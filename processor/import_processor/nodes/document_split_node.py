import re, os, json
from typing import Tuple, List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError
from knowledge.utils.markdown_util import MarkdownTableLinearizer

# 标题行正则（# {1,6}）：模块级预编译一次，避免每次切分文档都重新编译
# ()捕获组: group(1):#(1)#(6) group(2)标题的内容
_HEADING_LINE_RE = re.compile(r"^\s*(#{1,6})\s+(.+)")


class DocumentSplitNode(BaseNode):
    name = "document_split_node"

    def process(self,state:ImportGraphState)->ImportGraphState:
        """
        文档切分的核心入口
        Args:
            state: 初始状态

        Returns:
            ImportGraphState: 处理后的状态

        """

        # 1. 参数校验
        config = self.config
        md_content,file_title,max_content_length,min_content_length = self._validate_state(state, config)

        # 2. 切分（一级策略：根据md文档中的标题来切分）多个章节（章节：标题之间的内容）
        sections:List[Dict[str,Any]] = self._split_by_headings(md_content,file_title)


        # 3. 二次切分或者合并
        final_section = self._split_and_merge(sections, config.max_content_length, config.min_content_length)

        # 4. 组装成chunk对象
        final_chunks = self._assemble_chunks(final_section)

        # 5. 备份(观察)
        self._back_up(final_chunks, state)

        # 6. 更新state (chunks)
        state['chunks'] = final_chunks

        # 7. 返回
        return state


    def _validate_state(self, state:ImportGraphState,config)->Tuple[str, str, int, int]:
        self.log_step("step1","切分文档的参数校验以及获取...")

        # 1. 获取md_content
        md_content = state.get("md_content")

        # 1.1 内容缺失时直接抛错：None 会在后续 split(None) 崩溃，空文档也没有切分意义
        if not md_content or not isinstance(md_content, str):
            raise StateFieldError(node_name=self.name, field_name='md_content', expected_type=str)

        # 2. 统一换行符
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        # 3. 获取文件的标题
        # 3.1 标题缺失时直接抛错：None 会作为 title 进入 section，后续 len(None) 崩溃
        file_title = state.get('file_title')
        if not file_title or not isinstance(file_title, str):
            raise StateFieldError(node_name=self.name, field_name='file_title', expected_type=str)

        # 4. 校验最大值和最小值
        if config.max_content_length <= 0 or config.min_content_length <=0 or config.max_content_length < config.min_content_length:
            raise ValueError(f"最大内容长度和最小内容长度必须大于0且最大内容长度必须大于最小内容长度")

        return md_content, file_title, config.max_content_length, config.min_content_length

    def _split_by_headings(self, md_content: str, file_title: str) -> List[Dict[str, Any]]:

        """
        parent_title:封装的原因主要为了后面短section在合并的时候有一个判断标准（同源：同一个父标题）
        根据标题来切分（# {1,6}都有可能）
        Args:
            md_content: 切分的md
            file_title: 上传文档标题

        Returns:
         List[Dict]:切分后的多个章节
        """

        in_fence = False  # 是否在代码块内
        body_liens = []
        sections = []  # 最终收集到的章节对象
        current_title = ""
        hierarchy = [""] * 7  # （数组）存储所有标题内容（作为section的父标题使用） 标题层级追踪数组
        current_level = 0

        def _flush()->List[Dict[str,Any]]:
            """
            打包section
            {
            "body": "收集到的所有行"
            “title”:"当前内容的标题"
            "parent_title":当前内容的父标题（最麻烦）
            "file_title":文档标题（最简单）
            }
            Returns:
            如果current_title没有，body有 能进入打包成section,【也有意义】
            如果current_title有,body没有。也打包成section:在合并阶段可以保留上【可选：建议留下来】在后续合并阶段没有任何影响
            如果current_title有，body也有 能进入打包成section【一定留】
            如果current_title没有 body也没有 不会进入（不能打包）
            """

            # 1. 处理内容行
            body = "\n".join(body_liens)
            if current_title or body:
                parent_title = ""
                for i in range(current_level-1,0,-1):
                    if hierarchy[i]:  # 找父标题的时候 排除某一个位置的空值
                        parent_title = hierarchy[i]
                        break

                if not parent_title:
                    parent_title = current_title if current_title else file_title

                sections.append({
                    "body": body,
                    "title": current_title if current_title else file_title,  # 内容标题
                    "parent_title": parent_title,  # 内容父标题
                    "file_title": file_title,
                })

        # 1.根据\n切分md_content
        md_lines = md_content.split("\n")

        # 2. 使用模块级预编译的标题正则（从MD中找标题#{1,6}）

        # 3. 遍历切分后md_lines
        for md_line in md_lines:

            # 3.1 检测代码块边界（``` 或 ~~~）代码块要留下来
            if md_line.strip().startswith("```") or md_line.strip().startswith("~~~"):
                in_fence = not in_fence

            # 3.2 判断是否要走正则
            match = _HEADING_LINE_RE.match(md_line) if not in_fence else None


            # 3.3 判断math 是否有
            # 代表匹配到了标题而且一定是非代码块中的# 标题
            if match:

                # 将 body_liens中收集到的行封装到section对象
                _flush()

                current_title = md_line  # 当前标题
                level = len(match.group(1))  # 当前标题的层级（# {1,6}）
                current_level = level
                hierarchy[level] = current_title  # 写入操作

                for i in range(level + 1, 7):
                    hierarchy[i] = ""  # 下面的清空
                # 没有匹配到标题[普通行] 或者是代码块（加入）
            else:
                body_liens.append(md_line)

        # 循环结束后，刷新最后一个 section
        _flush()
        return sections

    def _split_and_merge(self, sections: List[Dict[str, Any]], max_content_length: int, min_content_length: int) -> \
            List[Dict[str, Any]]:
        """
        切分较大的章节（section）以及合并较小章节（section）
        Args:
            sections: 所有经过一级切分后的章节
            max_content_length: 最大内容长度（content:title+\n\n+body）如果section中的内容长度超过阈值max_content_length，就需要进行切割，反之不需要切割（尽量保证不要太多的section进行二次切割）
            min_content_length: 最小内容长度：如果section中的内容长度不足阈值min_content_length,就需要对该section进行合并。同源机制合并（section相同的父标题才合。尽量保证确实比较小的内容才合并，不要把大多数的section都合并）

        Returns:
            先切后合

        """

        # 1. 切分
        current_section = []
        for section in sections:
            current_section.extend(self._split_long_section(section, max_content_length))

        # 2. 合并（同时携带max_content_length：合并结果不得超过上限，否则切分努力白费）
        final_sections = self._merger_short_section(current_section, min_content_length, max_content_length)

        return final_sections

    def _split_long_section(self, section: Dict[str, Any], max_content_length: int) -> List[Dict[str, Any]]:
        """
        切分较大的章节（section）
        Args:
            section: 单个章节
            max_content_length: 最大内容长度（content:title+\n\n+body）如果section中的内容长度超过阈值max_content_length，就需要进行切割，反之不需要切割（尽量保证不要太多的section进行二次切割）

        Returns:
            切分后的章节

        """

        # 1. 获取section对象的属性
        body = section.get("body")
        title = section.get("title")
        parent_title = section.get("parent_title")
        file_title = section.get("file_title")

        if len(title) > 80:
            title = title[:80]   # 防御性编程：title 本身就超长的极端情况

        if "<table>" in body:
            self.logger.info("检查到section中有表格")
            body = MarkdownTableLinearizer.process(body)

        # 2. 获取标题前缀
        title_prefix = f"{title}\n\n"

        # 3. 获取总长度【标题(前缀)+body】
        total_length = len(title_prefix) + len(body)

        # 4. 判断总长度是否超过阈值
        if total_length <= max_content_length:
            return [section]


        # 5. 能切分的内容长度计算出来（body）
        body_length = max_content_length - len(title_prefix)
        if body_length <= 0:
            return [section]  # 防御性编程：title 本身就超长的极端情况

        # 6. 切分(6.1 用谁切:LangChain的切分器:递归切分器 6. 去切谁:body)
        # 6.1 切分器对象  # 1. chunk_size:chunk块的大小  2.chunk_overlap块与块之间的重叠的字符数  3. separators：分割符 【"\n\n",'\n',' ',''】
        text_spliter = RecursiveCharacterTextSplitter(
            chunk_size=body_length,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "？", "！", "；", ".", "?", "!", ';', " ", ""],
            keep_separator=True)

        # 6.2 切分器对象切分
        split_bodies = text_spliter.split_text(body)

        # 6.3 判断
        if len(split_bodies) == 1:
            return [section]

        # 6.4 遍历（遍历变量重命名：不能与参数 section 同名遮蔽）
        sub_sections = []
        for index, split_body in enumerate(split_bodies):
            sub_sections.append({
                "body": split_body,
                "title": f"{title}_{index + 1}",
                "parent_title": parent_title,
                "file_title": file_title
            })

        # 7. 返回
        return sub_sections

    def _merger_short_section(self, current_sections, min_content_length, max_content_length) -> List[Dict[str, Any]]:
        """
        合并短的章节：
        短章节来源：
        来源1：原本根据一级标题切分之后可能内容就很短
        来源2：（LangChain递归切分器）二次切分之后可能有很短的内容
        合并策略：
        条件1：section很短比最小阈值还小
        条件2：同源（父标题相同）
        条件3：合并后不得超过max_content_length（短section连续合并不复查上限，
                可能拼出一个远超上限的巨型chunk，让前面的切分努力白费）
        Args:
            current_sections: 二次切分后的所有section对象
            min_content_length:每一个section的内容最小长度
            max_content_length:每一个section的内容最大长度

        Returns:
            合并之后的section对象

            贪心累加算法

        """

        # 0. 防御：切分结果为空（空文档等极端情况）直接返回，避免 current_sections[0] 越界
        if not current_sections:
            return []

        current_section = current_sections[0]
        final_sections = []

        # 2. 遍历合并
        for next_section in current_sections[1:]:
            same_parent = (current_section['parent_title'] == next_section['parent_title'])
            current_body = current_section.get('body') or ""
            next_body = next_section.get('body') or ""
            # 合并后的预估长度：两段body + 连接符\n\n
            merged_would_exceed = (len(current_body) + len(next_body) + 2) > max_content_length
            if same_parent and len(current_body) < min_content_length and not merged_would_exceed:
                # 合并 body
                current_section['body'] = current_body.rstrip() + "\n\n" + next_body.lstrip()
                # 标题回退为父标题
                current_section['title'] = current_section['parent_title']
            else:
                # 封箱
                final_sections.append(current_section)
                current_section = next_section  # 更新指针
        # 最后一个封箱
        final_sections.append(current_section)
        return final_sections

    def _assemble_chunks(self, final_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        组装最后的chunks
        Args:
            final_section:

        Returns:

        """
        final_chunks = []
        for section in final_sections:
            body = section.get('body')
            title = section.get('title')
            parent_title = section.get('parent_title')
            file_title = section.get('file_title')

            content = f"{title}\n\n{body}"

            final_chunks.append({
                "content": content,
                "title": title,
                "parent_title": parent_title,
                "file_title": file_title
            })
        self.logger.info(f"最终切割后能够进入到嵌入节点的chunk个数:{len(final_chunks)}")
        return final_chunks

    def _back_up(self, final_chunks, state: ImportGraphState):
        """将切分结果备份到 JSON 文件"""
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return
        try:
            os.makedirs(local_dir, exist_ok=True)  # 如果目录存在 不报错
            output_path = os.path.join(local_dir, "chunks.json")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_chunks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


if __name__ == '__main__':
    document_split_node = DocumentSplitNode()

    md_path = r"C:\Users\a3384310973\Desktop\BJ251208\shopkeeper_brain\knowledge\processor\import_processor\nodes\temp\output\HUAWEI MateStation B515 用户手册-(PanguL,Windows11_02,zh-cn)\HUAWEI MateStation B515 用户手册-(PanguL,Windows11_02,zh-cn).md"

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    init_state = {
        "md_content": md_content,
        "file_title": "HUAWEI MateStation B515 用户手册",
        "file_dir": r"C:\Users\a3384310973\Desktop\BJ251208\shopkeeper_brain\knowledge\processor\import_processor\nodes\temp"
    }
    document_split_node.process(init_state)

    print(init_state)