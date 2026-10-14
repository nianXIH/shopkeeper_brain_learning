from pathlib import Path

from knowledge.processor.import_processor.base import BaseNode, setup_logging,T
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError,ValidationError

class EntryNode(BaseNode):
    name = "entry_node"
    def process(self,state:ImportGraphState)->ImportGraphState:
        """
        根据上传文件的后缀 修改state中 is_md_read_enabled  is_pdf_read_enabled
        Args:
            state: 初始状态

        Returns:
            修改后的状态
        """

        # 1. 获取上下文的文件
        import_file_path = state.get('import_file_path','')
        file_dir = state.get('file_dir','')

        # 2. 判断
        if not import_file_path:
            raise  StateFieldError(node_name=self.name,field_name='import_file_path',expected_type=str)
        if not file_dir:
            raise  StateFieldError(node_name=self.name,field_name='file_dir',expected_type=str)

        # 3.Path 的标准化
        import_file_path_obj = Path(import_file_path)
        file_dir_obj = Path(file_dir)

        # 4.判断
        if not import_file_path_obj.exists():
            raise StateFieldError(node_name=self.name, field_name='import_file_path_obj', expected_type=Path)
        if not file_dir_obj.exists():
            raise StateFieldError(node_name=self.name, field_name='file_dir_obj', expected_type=Path)

        # 5. 获取文件的后缀
        file_suffix = import_file_path_obj.suffix
        if file_suffix == '.pdf':
            state['is_pdf_read_enabled'] = True
            state['is_md_read_enabled'] = False
            state['pdf_path'] = str(import_file_path_obj)
        elif file_suffix == '.md':
            state['is_md_read_enabled'] = True
            state['is_pdf_read_enabled'] = False
            state['md_path'] = str(import_file_path_obj)
        else:
            self.logger.error(f"该文件后缀名{file_suffix}不支持")
            raise ValidationError(message=f"该文件的后缀名{file_suffix}不支持",node_name=self.name)

        # 6. 获取上传文件的标题，更新到state中
        state['file_title'] = import_file_path_obj.stem

        # 7. 返回state
        return state


if __name__ == '__main__':
    entry_node = EntryNode()
    init_state = {
        "import_file_path": r"C:\Users\a3384310973\Desktop\BJ251208\shopkeeper_brain\knowledge\processor\import_processor\nodes\temp\HUAWEI MateStation B515 用户手册-(PanguL,Windows11_02,zh-cn).pdf",
        "file_dir": r"C:\Users\a3384310973\Desktop\BJ251208\shopkeeper_brain\knowledge\processor\import_processor\nodes\temp"
    }
    result = entry_node.process(init_state)

    import json

    print(json.dumps(result, ensure_ascii=False, indent=4))
