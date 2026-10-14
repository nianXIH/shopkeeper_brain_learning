import subprocess
import sys
import shutil
import time
import zipfile
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_processor.base import BaseNode, T, setup_logging
from knowledge.processor.import_processor.exceptions import StateFieldError, PdfConversionError
from knowledge.processor.import_processor.state import ImportGraphState


class PdfToMdNode(BaseNode):
    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点的处理逻辑入口
        :param state:
        :return:
        """

        # 1.获取导入文件的路径以及输出目录

        import_file_path_obj ,file_dir_obj = self._validate_state(state)

        # 2. 执行mineru解析（命令： mineru -p input_path -o output_dir --source=local）
        processed_code = self._execute_mineru_parse(import_file_path_obj, file_dir_obj)
        if processed_code != 0:
            raise PdfConversionError(message="MinerU解析PDF失败", node_name=self.name)

        # 3. 获取解析后md_path
        md_path = self.get_md_path(import_file_path_obj, file_dir_obj)

        # 4. 更新state["md_path"]
        state['md_path'] = md_path

        # 5. 返回state
        return state

    def _validate_state(self,state:ImportGraphState)->Tuple[Path,Path]:
        """

        :param state: 导入图谱节点状态
        :return: 导入文件的路径以及输出目录
        Tuple[Path,Path]
        """

        self.log_step("step1", "准备校验和获取解析文件路径和输出目录")

        # 1. 获取解析的文件path
        import_file_path = state.get('import_file_path', '')

        # 2. 判断是否为空
        if not import_file_path:
            raise StateFieldError(node_name=self.name, field_name='import_file_path', expected_type=str)

        # 3. 标准化解析文件的路径
        import_file_path_obj = Path(import_file_path)

        # 4. 判断是否是一个有效的路径
        if not import_file_path_obj.exists():
            raise StateFieldError(node_name=self.name, field_name='import_file_path', expected_type=str,
                                  message="解析文件的路径不存在")

        # 5. 获取输出文件目录
        file_dir = state.get('file_dir', '')

        # 6. 判断输出文件目录
        if not file_dir:
            # 6.1 获取导入文件的目录（默认值）
            file_dir = import_file_path_obj.parent

        # 7. 标准化输出目录
        file_dir_obj = Path(file_dir)

        # 8. 判断是否是一个有效的目录，不存在则抛出异常
        if not file_dir_obj.exists():
            raise StateFieldError(node_name=self.name, field_name='file_dir', expected_type=str,
                                  message="输出文件目录不存在")

        self.logger.info(f"解析的文件路径{import_file_path}")
        self.logger.info(f"输出的文件目录{file_dir}")

        # 9. 返回校验通过的路径对象
        return import_file_path_obj,file_dir_obj

    def _execute_mineru_parse(self, import_file_path_obj: Path,
                              file_dir_obj: Path) -> int:
        """

        :param import_file_path_obj: 解析文件的path路径
        :param file_dir_obj:  解析后的文件输出目录
        :return:  状态[0或者非0]
        0:成功的（底层封装状态码）
        非0：失败的
        """
        # mineru -p input_path -o output_dir --source=local

        # 1. 定义cmd
        # 关键点：
        #   ① MinerU 4.x 中 `mineru parse` 是「文档库/Agent 阅读」工作流，图片只会输出成
        #      定位符(doc:.../page:x/block:y)，不落地；必须用 `mineru-kit parse`。
        #   ② `mineru-kit parse --format markdown` 会把图片内联成 base64，同样不生成 images/ 目录；
        #      只有 `--format zip` 才会打包出 images/ 独立图片文件，因此这里用 zip，随后再解压。
        file_stem = import_file_path_obj.stem
        zip_output_dir = file_dir_obj / "output"
        zip_output_dir.mkdir(parents=True, exist_ok=True)
        print("import_file_path_obj:", import_file_path_obj)
        print("file_dir_obj:", file_dir_obj)
        print("zip_output_dir:", zip_output_dir)

        cmd = [
            self._resolve_mineru_exe(),
            "parse",
            str(import_file_path_obj),
            "--pages", "all",
            "-o",
            str(zip_output_dir),
            "--tier", "standard",
            "--format", "zip"
        ]

        # 2. 利用子进程执行cmd命令(子进程解析的日志【正常日志和错误日志都要】)
        # 子进程（执行命令产生的日志[正常、错误日志]）-----管子----外部线程（_execute_mineru_parse）
        start_time = time.time()
        proc = subprocess.Popen(args=cmd,
                                stdout=subprocess.PIPE,  # 接收正确日志
                                stderr=subprocess.STDOUT,  # 接收错误日志
                                text=True,  # 输出二进制字节流，输出字符串
                                errors="replace",  # 特殊的字符码替换成?、菱形
                                encoding="utf-8",  # utf-8进行解码
                                bufsize=1  # 实时输出。按行输出遇到\n换行符 就将日志产生出来
                                )

        # 3. 实时打印日志
        if proc.stdout:
            for line in proc.stdout:
                self.logger.info(f"MinerU解析产生的日志：{line}")

        # 4. 主线程等待子进程做完（状态码0：成功 反之失败）
        processed_result = proc.wait()
        end_time = time.time()
        if processed_result != 0:
            self.logger.info(f"MinerU解析失败")
            return processed_result

        self.logger.info(f"MinerU解析成功，耗时{end_time - start_time}秒")

        # 5. 解压 zip，得到 <output>/<文件名>/markdown.md 与同级的 images/ 目录
        zip_path = zip_output_dir / f"{file_stem}.zip"
        if not zip_path.exists():
            self.logger.error(f"未找到MinerU解析结果压缩包：{zip_path}")
            return 1

        extract_dir = self._get_extract_dir(import_file_path_obj, file_dir_obj)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        self.logger.info(f"已解压MinerU解析结果到：{extract_dir}")

        # 6. zip 内 md 固定名为 markdown.md，重命名为 <原文件名>.md，
        #    便于下游节点用 md 文件名作为文档标题
        extracted_md = extract_dir / "markdown.md"
        final_md = extract_dir / f"{file_stem}.md"
        if extracted_md.exists():
            if final_md.exists():
                final_md.unlink()
            extracted_md.rename(final_md)

        # 7. 清理中间产物 zip
        zip_path.unlink(missing_ok=True)

        return 0

    def _resolve_mineru_exe(self) -> str:
        """
        定位 mineru-kit 可执行文件。
        优先取当前 Python 解释器同级目录下的可执行文件（venv 未激活、Scripts 不在 PATH 时也能找到），
        找不到再回退到 PATH 中的 mineru-kit。
        """
        exe_name = "mineru-kit.exe" if sys.platform == "win32" else "mineru-kit"
        candidate = Path(sys.executable).parent / exe_name
        if candidate.exists():
            return str(candidate)
        which = shutil.which("mineru-kit")
        if which:
            return which
        # 最后兜底，交给系统在 PATH 中查找
        return "mineru-kit"

    def _get_extract_dir(self, import_file_path_obj: Path, file_dir_obj: Path) -> Path:
        """
        解析结果(zip)的解压目录：file_dir/output/<文件名>/
        解压后该目录下为 markdown.md(会被重命名为 <文件名>.md) 与同级的 images/ 目录
        """
        return file_dir_obj / "output" / import_file_path_obj.stem
    
    def get_md_path(self, import_file_path_obj: Path, file_dir_obj: Path) -> str:
        """
        获取解析后md的路径
        :param import_file_path_obj:
        :param file_dir_obj:
        :return:
        md_path= <file_dir>/output/<文件名>/<文件名>.md（与 images/ 目录同级）
    
        Path:吉祥三宝：name:全名[文件名字.后缀]   stem【文件名字，没有后缀】    suffix【文件的后缀】
        """
        file_name = import_file_path_obj.stem
    
        return str(self._get_extract_dir(import_file_path_obj, file_dir_obj) / f"{file_name}.md")


##########################################
# 测试
##########################################

if __name__ == '__main__':
    setup_logging()
    # 1. 构建节点实例
    pdf_to_md_node = PdfToMdNode()

    # 2. 构建该节点状态
    init_state = {
        "import_file_path": r"C:\Users\a3384310973\Desktop\BJ251208\shopkeeper_brain\knowledge\processor\import_processor\nodes\temp\HUAWEI MateStation B515 用户手册-(PanguL,Windows11_02,zh-cn).pdf",
        "file_dir": r"C:\Users\a3384310973\Desktop\BJ251208\shopkeeper_brain\knowledge\processor\import_processor\nodes\temp"
    }

    # 3. 直接调用process
    result = pdf_to_md_node.process(init_state)

    # 4. 序列化（将对象转成字符串） 反序列化（将字符串转成对象）
    import json
    result_str = json.dumps(result, indent=4, ensure_ascii=False)
    print(result_str)
