from pathlib import Path
from typing import Any
from functools import cached_property
from fstdb import TreeNode

from .rm.db import Record as GCHRecord
from .rm.cfg import CustomConfigManagerFactory, DBReferHandler, ConfigManager
from .rm.db import DB, WorkDBFactory, TaskDBFactory, DBFactory
from .work_context import WorkContext
from .tools.command_manager import CommandManager
from .rm.metadata import resolve_record_metadata


class RMFactory:
    def __init__(self):
        pass

    @property
    def db_root_dir_path(self)->Path:
        return Path("/home/resources")

    @cached_property
    def work_db_factory(self)->DBFactory:
        return WorkDBFactory(TreeNode, DB, GCHRecord)

    @cached_property
    def task_db_factory(self)->DBFactory:
        return TaskDBFactory(TreeNode, DB, GCHRecord)

    @cached_property
    def work_db(self)->DB:
        return self.work_db_factory.create_tree_db(self.db_root_dir_path)

    @cached_property
    def db_config_refer_handler(self)->DBReferHandler:
        return DBReferHandler(work_db=self.work_db)

    @cached_property
    def config_manager_factory(self)->CustomConfigManagerFactory:
        return CustomConfigManagerFactory(db_refer_handler=self.db_config_refer_handler)

    @cached_property
    def config_manager(self)->ConfigManager:
        return self.config_manager_factory.config_manager

    def get_command_manager(self)->CommandManager:
        return CommandManager(dir_path=Path("/home"))

    def get_work_context(self, work_id)->WorkContext:
        return WorkContext(
            task_db_factory=self.task_db_factory,
            work_record=self.work_db.get_record(work_id),
            config_manager=self.config_manager,
        )


work_db = RMFactory().work_db
config_manager = RMFactory().config_manager


def _record(id:int)->GCHRecord:
    return work_db.get_record(id)


def _config(id:int|str)->Any:
    if isinstance(id, int):
        path = _record(id).config_file_path
    else:
        path = id
    return config_manager.load_config(path, handling=True, lazy_handling=False)


def _prop(id:int)->Any:
    record = _record(id)
    return resolve_record_metadata(
        _config(id),
        record.prop_file_path,
        config_manager,
        handling=True,
        lazy_handling=False,
    )


def _path(id:int)->Path:
    return work_db.get_record(id).path
