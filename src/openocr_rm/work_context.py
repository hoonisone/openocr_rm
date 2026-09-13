from .rm.cfg import ConfigManager
from .rm.db import TaskDBFactory, Record as GCHRecord
from .rm.metadata import resolve_record_metadata
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from functools import cached_property
from textwrap import dedent
import shutil

class WorkContext:
    TRAIN_TASK_ID = 1


    def __init__(self,
            task_db_factory:TaskDBFactory,
            work_record:GCHRecord,
            config_manager:ConfigManager,
        ):
        self.task_db_factory = task_db_factory
        self.work_record = work_record
        self.config_manager = config_manager


    @property
    def id(self):
        return self.work_record.id


    @cached_property
    def path(self)->Path:
        return self.work_record.path
    
    @cached_property
    def task_db(self):
        return self.task_db_factory.create_tree_db(self.path)

    @cached_property
    def prop_path(self)->Path:
        return self.path / "prop.py"

    @cached_property
    def metadata(self):
        return resolve_record_metadata(
            self.config,
            self.prop_path,
            self.config_manager,
            handling=True,
            lazy_handling=False,
        )

    @property
    def eval_dataset_ids(self):
        return self.metadata["eval_dataset_ids"]

    @property
    def eval_epochs(self):
        return self.metadata.get("eval_epochs", [])

    @property
    def eval_steps(self):
        return self.metadata.get("eval_steps", [])

    @cached_property
    def config_path(self)->Path:
        return self.work_record.config_file_path

    @property
    def config(self)->dict:
        return self.config_manager.load_config(self.config_path, handling=True, lazy_handling=False)

    def _invalidate_task_cache(self)->None:
        self.__dict__.pop("all_tasks", None)

    def _context_for_record(self, record: GCHRecord) -> Union['TrainTaskContext', 'EvalTaskContext']:
        peek = TaskContext(
            work_record=self.work_record,
            task_record=record,
            config_manager=self.config_manager,
        )
        task_type = peek.metadata['type']
        if task_type == "train":
            return TrainTaskContext(
                work_record=self.work_record,
                task_record=record,
                config_manager=self.config_manager,
            )
        if task_type == "test":
            return EvalTaskContext(
                work_record=self.work_record,
                task_record=record,
                config_manager=self.config_manager,
            )
        raise ValueError(f"Invalid task type: {task_type}")

    def get_task(self, task_id) -> Optional[Union['TrainTaskContext', 'EvalTaskContext']]:
        if task_id not in self.task_db:
            return None
        return self._context_for_record(self.task_db.get_record(task_id))

    def train_task(self) -> Optional['TrainTaskContext']:
        if self.TRAIN_TASK_ID not in self.task_db:
            return None
        context = self.get_task(self.TRAIN_TASK_ID)
        if context is None:
            return None
        if not isinstance(context, TrainTaskContext):
            raise ValueError(f"Task {self.TRAIN_TASK_ID} is not a train task")
        return context

    def eval_task(self, dataset_id, *, epoch=None, step=None) -> Optional['EvalTaskContext']:
        mode, value = self._resolve_eval_point(epoch=epoch, step=step)
        try:
            return self.all_tasks["test"][dataset_id][mode][value]
        except KeyError:
            return None

    def make_train_task(self) -> 'TrainTaskContext':
        existing = self.train_task()
        if existing is not None:
            return existing
        record = self.task_db.create_record("train")
        assert record.id == self.TRAIN_TASK_ID, (
            "Train task id is not " + str(self.TRAIN_TASK_ID)
            + " This means that other tasks were created before the train task"
        )
        context = TrainTaskContext(
            work_record=self.work_record,
            task_record=record,
            config_manager=self.config_manager,
        )
        context.initialize()
        self._invalidate_task_cache()
        return context

    def remove_train_task(self, dry_run: bool = False)->None:
        train_task = self.train_task()
        if train_task is None:
            return
        if dry_run:
            print(train_task.path)
        else:
            shutil.rmtree(train_task.path)
            self._invalidate_task_cache()

    def remove_all_tasks(self, dry_run: bool = False)->None:
        for task_id in list(self.task_db.ids):
            path = self.task_db.get_record(task_id).path
            if dry_run:
                print(path)
            else:
                shutil.rmtree(path)
        if not dry_run:
            self._invalidate_task_cache()

    def remove_all_eval_tasks(self, dry_run: bool = False)->None:
        for task_id in list(self.task_db.ids):
            context = self.get_task(task_id)
            if not isinstance(context, EvalTaskContext):
                continue
            if dry_run:
                print(context.path)
            else:
                shutil.rmtree(context.path)
        if not dry_run:
            test_dir = self.path / "test"
            if test_dir.exists():
                shutil.rmtree(test_dir)
            self._invalidate_task_cache()

    @cached_property
    def all_tasks(self)->Dict:
        tasks = {}
        for task_id in self.task_db.ids:
            context = self.get_task(task_id)
            if isinstance(context, TrainTaskContext):
                assert "train" not in tasks, "Train task already exists"
                tasks["train"] = context
            elif isinstance(context, EvalTaskContext):
                mode, value = context.eval_point
                tasks.setdefault("test", {}).setdefault(context.dataset_id, {}).setdefault(
                    mode, {})[value] = context
            else:
                raise ValueError(f"Invalid task type for id {task_id}")
        return tasks


    def _resolve_eval_point(self, epoch=None, step=None):
        if (epoch is None) == (step is None):
            raise ValueError("Exactly one of epoch or step must be provided.")
        if epoch is not None:
            return "epoch", int(epoch)
        return "step", int(step)

    def is_eval_point_in_metadata(self, epoch=None, step=None) -> bool:
        mode, value = self._resolve_eval_point(epoch=epoch, step=step)
        if mode == "epoch":
            return value in self.eval_epochs
        return value in self.eval_steps

    def make_eval_record_name(self, eval_dataset_id, epoch=None, step=None)->str:
        mode, value = self._resolve_eval_point(epoch=epoch, step=step)
        if mode == "epoch":
            return f"test/dataset_{eval_dataset_id}/epoch_{value}"
        return f"test/dataset_{eval_dataset_id}/iter_{value}"

    def has_eval_checkpoint(
        self,
        epoch=None,
        step=None,
        train_ctx: Optional['TrainTaskContext'] = None,
    ) -> bool:
        if train_ctx is None:
            train_ctx = self.train_task()
        if train_ctx is None:
            return False
        mode, value = self._resolve_eval_point(epoch=epoch, step=step)
        if mode == "epoch":
            return train_ctx.does_weight_exists(value)
        return train_ctx.does_step_weight_exists(value)

    def eval_tasks(self) -> List['EvalTaskContext']:
        tasks = []
        for task_id in self.task_db.ids:
            context = self.get_task(task_id)
            if isinstance(context, EvalTaskContext):
                tasks.append(context)
        return tasks

    def make_eval_task(
        self,
        eval_dataset_id,
        epoch=None,
        step=None,
        check_weight: bool = False,
        train_ctx: Optional['TrainTaskContext'] = None,
    ) -> Optional['EvalTaskContext']:
        """
            check_weight가 True이면 해당 epoch/step weight가 있을 때만 새 task를 생성한다.
            이미 task가 있으면 weight 여부와 관계없이 반환한다.
            train_ctx가 넘어오면 weight 체크에 재사용하고, 없으면 필요할 때 조회한다.
        """
        mode, value = self._resolve_eval_point(epoch=epoch, step=step)
        epoch_arg = value if mode == "epoch" else None
        step_arg = value if mode == "step" else None
        existing = self.eval_task(eval_dataset_id, epoch=epoch_arg, step=step_arg)
        if existing is not None:
            return existing
        if check_weight:
            if train_ctx is None:
                train_ctx = self.train_task()
            if train_ctx is None:
                return None
            if not self.has_eval_checkpoint(
                epoch=epoch_arg,
                step=step_arg,
                train_ctx=train_ctx,
            ):
                return None
        record_name = self.make_eval_record_name(
            eval_dataset_id,
            epoch=epoch_arg,
            step=step_arg,
        )
        record = self.task_db.create_record(record_name)
        eval_context = EvalTaskContext(
            work_record=self.work_record,
            task_record=record,
            config_manager=self.config_manager,
        )
        eval_context.initialize(
            eval_dataset_id,
            epoch=epoch_arg,
            step=step_arg,
        )
        self._invalidate_task_cache()
        return eval_context


    def make_all_eval_tasks(self, check_weight: bool = True)->None:
        train_ctx = self.train_task() if check_weight else None
        for dataset_id in self.eval_dataset_ids:
            for epoch in self.eval_epochs:
                self.make_eval_task(
                    dataset_id,
                    epoch=epoch,
                    check_weight=check_weight,
                    train_ctx=train_ctx,
                )
            for step in self.eval_steps:
                self.make_eval_task(
                    dataset_id,
                    step=step,
                    check_weight=check_weight,
                    train_ctx=train_ctx,
                )




    @property
    def all_eval_checkpoints_exist(self) -> bool:
        train_ctx = self.train_task()
        if train_ctx is None:
            return False
        for epoch in self.eval_epochs:
            if not self.has_eval_checkpoint(epoch=epoch, train_ctx=train_ctx):
                return False
        for step in self.eval_steps:
            if not self.has_eval_checkpoint(step=step, train_ctx=train_ctx):
                return False
        return True


class TaskContext:
    def __init__(self, 
        work_record:GCHRecord,
        task_record:GCHRecord,
        config_manager:ConfigManager
    ):
        self.work_record = work_record
        self.task_record = task_record
        self.config_manager = config_manager

    @property
    def id(self):
        return self.task_record.id

    @cached_property
    def path(self)->Path:
        return self.task_record.path

    def get_task_path(self)->GCHRecord:
        return self.task_record

    @cached_property
    def config_path(self)->Path:
        return self.task_record.path / "config.py"
    
    @cached_property
    def prop_path(self)->Path:
        return self.task_record.path / "prop.py"

    @cached_property
    def config(self):
        try:
            return self.config_manager.load_config(self.config_path, handling=True, lazy_handling=True)
        except Exception as e:
            # print(f"Error loading config: {e}")
            print(f"Config path: {self.config_path}")
            raise ValueError(f"Error loading config: {e}") from e


    @cached_property
    def metadata(self):
        return resolve_record_metadata(
            self.config,
            self.prop_path,
            self.config_manager,
            handling=True,
            lazy_handling=False,
        )

    def write_config(self, config:str)->None:
        with open(self.config_path, "w") as f:
            f.write(config)


    def write_prop(self, text:str)->None:
        with open(self.prop_path, "w") as f:
            f.write(text)




class TrainTaskContext(TaskContext):
    def initialize(self)->None:
        config = dedent(f"""

            from openocr_rm import _path, _config
            from datetime import datetime as _datetime
            from pathlib import Path as _Path

            work_id = {self.work_record.id}
            _base = _config(work_id)

            Metadata = dict(
                type="train",
                epoch=None,
            )

            Wandb = dict(
                project="OpenOCR-GCH",
                name=str(work_id),
                sync_tensorboard=True,
            )

            """+"""
            _weights_dir = f"{__file__}/../weights"
            _output_dir = f"{__file__}/../{_datetime.now().strftime('%Y%m%d_%H%M%S')}"
            _save_res_path = f"{_output_dir }/output/res.txt"
            _pretrained_model = (_Path(_weights_dir)/"pretrained.pth").resolve()
            _pretrained_model = _pretrained_model.as_posix() if _pretrained_model.exists() else None
            _checkpoints = (_Path(_weights_dir)/"latest.pth").resolve()
            _checkpoints = _checkpoints.as_posix() if _checkpoints.exists() else None

            _train_data_dir_list = [_config(id)["data_dir"] for id in _base["Metadata"]["train_dataset_ids"]]
            _train_label_file_list = [_config(id)["label_path"] for id in _base["Metadata"]["train_dataset_ids"]]
            _eval_data_dir_list = [_config(id)["data_dir"] for id in _base["Metadata"]["eval_dataset_ids"]]
            _eval_label_file_list = [_config(id)["label_path"] for id in _base["Metadata"]["eval_dataset_ids"]]
        """)
        self.write_config(config)

    # def make_train_command(self)->str:
    #     return f"python /home/src/atomicstr/openocr/tools/train_rec.py --work_id {self.work_record.id} --task_id {self.task_record.id}"
    

    def does_weight_exists(self, epoch:int)->bool:
        return self.weight_path(epoch).exists()

    def does_step_weight_exists(self, step:int)->bool:
        return self.step_weight_path(step).exists()


    @cached_property
    def weight_dir(self)->Path:
        return self.task_record.path / "weights"

    def weight_path(self, epoch:int)->Path:
        return self.weight_dir / f"epoch_{epoch}.pth"

    def step_weight_path(self, step:int)->Path:
        return self.weight_dir / f"iter_{step}.pth"

    @cached_property
    def weight_epoches(self)->List[int]:
        paths = self.weight_dir.rglob("*.pth")
        epochs =  [self.extract_epoch_from_weight_path(path) for path in paths]
        epochs = [epoch for epoch in epochs if epoch is not None]
        return epochs


    def extract_epoch_from_weight_path(self, weight_path:Path)->int|None:
        try:
            return int(weight_path.stem.split("_")[-1])
        except:
            return None

    @cached_property
    def is_completed(self)->bool:
        epoch_num = self.config['Global']['epoch_num']
        return self.does_weight_exists(epoch_num)


class EvalTaskContext(TaskContext):

    @property
    def dataset_id(self):
        dataset_id = self.metadata.get('dataset_id')
        if dataset_id is None:
            raise ValueError("Eval task must have dataset id in metadata, but it's not set")
        return dataset_id

    @property
    def eval_point(self):
        epoch = self.metadata.get('epoch', None)
        step = self.metadata.get('step', None)
        if (epoch is None) == (step is None):
            raise ValueError(
                "Eval task metadata must contain exactly one of epoch or step."
            )
        if epoch is not None:
            return "epoch", int(epoch)
        return "step", int(step)

    def initialize(self, eval_dataset_id, epoch=None, step=None)->None:
        if (epoch is None) == (step is None):
            raise ValueError("Exactly one of epoch or step must be provided.")
        if epoch is not None:
            weight_file = f"epoch_{int(epoch)}.pth"
            selector_metadata = f"epoch={int(epoch)}"
        else:
            weight_file = f"iter_{int(step)}.pth"
            selector_metadata = f"step={int(step)}"

        config = dedent(f"""
                from openocr_rm import _path, _config
                from datetime import datetime as _datetime

                work_id = {self.work_record.id}
                _base = _config(work_id)

                _weights_dir = _path(work_id)/"train___id_1/weights"
                _pretrained_model = _weights_dir/"{weight_file}"

                Metadata = dict(
                    type="test",
                    dataset_id={eval_dataset_id},
                    {selector_metadata},
                )
                """+"""
                _output_dir = f"{__file__}/../{_datetime.now().strftime('%Y%m%d_%H%M%S')}"
                _save_res_path = f"{_output_dir }/output/res.txt"

                _train_data_dir_list = None
                _train_label_file_list = None
                _eval_data_dir_list = [_config(Metadata["dataset_id"])["data_dir"]]
                _eval_label_file_list = [_config(Metadata["dataset_id"])["label_path"]]
            """)

        self.write_config(config)

    # def make_eval_command(self)->str:
    #     return f"python /home/src/atomicstr/openocr/tools/eval_rec.py --work_id {self.work_record.id} --task_id {self.task_record.id}"

    @cached_property
    def eval_result_path(self)->Path:
        return self.task_record.path / "eval_result.yml"

    def eval_result(self)->Optional[str]:
        import yaml

        path = self.eval_result_path
        
        with open(path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)


    def save_eval_result(self, result:dict)->None:
        import yaml
        path = self.eval_result_path
        with open(path, "w") as f:
            yaml.dump(result, f, allow_unicode=True)

            

    def is_evaluated(self)->bool:
        return self.eval_result_path.exists()
        #     return False
        # if self.eval_result_path is None:
        #     return False
        # # if len(self.eval_result()) == 0:
        # #     return False
        # return True