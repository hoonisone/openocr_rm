from .rm.cfg import ConfigManager
from .rm.db import TaskDBFactory, Record as GCHRecord
from .rm.metadata import resolve_record_metadata
from pathlib import Path
from typing import Optional, Dict, Any, List

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

    @cached_property
    def config_path(self)->Path:
        return self.work_record.config_file_path

    @property
    def config(self)->dict:
        return self.config_manager.load_config(self.config_path, handling=True, lazy_handling=False)

    



    def get_train_task_context(self, task_record)->'TrainTaskContext':
        
        return TrainTaskContext(
            work_record=self.work_record,
            task_record=task_record, 
            config_manager=self.config_manager
        )

    def get_eval_task_context(self, task_record)->'EvalTaskContext':
        
        return EvalTaskContext(
            work_record=self.work_record,
            task_record=task_record, 
            config_manager=self.config_manager
        )

    def get_train_task_record(self)->Optional[GCHRecord]:
        if self.TRAIN_TASK_ID in self.task_db:
            return self.task_db.get_record(self.TRAIN_TASK_ID)
        else:
            return None

    

    def make_train_task(self)->GCHRecord:
        train_task = self.get_train_task_record()
        if train_task is None:
            train_task = self.task_db.create_record("train")
            assert train_task.id == self.TRAIN_TASK_ID, "Train task id is not " + str(self.TRAIN_TASK_ID) + " This means that other tasks were created before the train task"
            self.get_train_task_context(train_task).initialize()
        return train_task


    def remove_train_task(self, dry_run: bool = False)->None:
        train_task = self.get_train_task_record()
        if train_task is not None:
            if dry_run:
                print(train_task.path)
            else:
                shutil.rmtree(train_task.path)

    def remove_all_tasks(self, dry_run: bool = False)->None:
        for task_id in list(self.task_db.ids):
            path = self.task_db.get_record(task_id).path
            if dry_run:
                print(path)
            else:
                shutil.rmtree(path)
        if not dry_run:
            self.__dict__.pop("all_tasks", None)

    def remove_all_eval_tasks(self, dry_run: bool = False)->None:
        for task_id in list(self.task_db.ids):
            if task_id == self.TRAIN_TASK_ID:
                continue
            record = self.task_db.get_record(task_id)
            task_type = self.get_eval_task_context(record).metadata.get("type")
            if task_type != "test":
                continue
            if dry_run:
                print(record.path)
            else:
                shutil.rmtree(record.path)
        if not dry_run:
            test_dir = self.path / "test"
            if test_dir.exists():
                shutil.rmtree(test_dir)
            self.__dict__.pop("all_tasks", None)

    @cached_property
    def all_tasks(self)->Dict:
        tasks = {}
        for id in self.task_db.ids:
            record = self.task_db.get_record(id)
            context = self.get_train_task_context(record)
            metadata = context.metadata
            task_type = metadata['type']
            dataset_id = metadata.get('dataset_id', None)
            epoch = metadata.get('epoch', None)
            step = metadata.get('step', None)


            if task_type == "train":
                assert "train" not in tasks, "Train task already exists"
                tasks["train"] = record
            elif task_type == "test":
                assert dataset_id is not None, "Eval task must have dataset id in metadata, but it's not set"
                assert (epoch is None) != (step is None), (
                    "Eval task must have exactly one of epoch or step in metadata."
                )
                if epoch is not None:
                    tasks.setdefault("test", {}).setdefault(dataset_id, {}).setdefault(
                        "epoch", {})[epoch] = record
                else:
                    tasks.setdefault("test", {}).setdefault(dataset_id, {}).setdefault(
                        "step", {})[step] = record
            else:
                raise ValueError(f"Invalid task type: {task_type}")
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
            return value in self.metadata.get("eval_epochs", [])
        return value in self.metadata.get("eval_steps", [])

    def make_eval_record_name(self, eval_dataset_id, epoch=None, step=None)->str:
        mode, value = self._resolve_eval_point(epoch=epoch, step=step)
        if mode == "epoch":
            return f"test/dataset_{eval_dataset_id}/epoch_{value}"
        return f"test/dataset_{eval_dataset_id}/iter_{value}"

    def get_eval_task_record(self, eval_dataset_id, epoch=None, step=None)->Optional[GCHRecord]:
        """
            task_record가 존재하면 반환
            없으면 None 반환
        """
        mode, value = self._resolve_eval_point(epoch=epoch, step=step)
        all_tasks = self.all_tasks
        if "test" not in all_tasks:
            return None
        if eval_dataset_id not in all_tasks["test"]:
            return None
        dataset_tasks = all_tasks["test"][eval_dataset_id]
        if mode not in dataset_tasks:
            return None
        if value not in dataset_tasks[mode]:
            return None
        return dataset_tasks[mode][value]
        
    def _get_train_task_context(self)->Optional['TrainTaskContext']:
        train_record = self.get_train_task_record()
        if train_record is None:
            return None
        return self.get_train_task_context(train_record)

    def _eval_checkpoint_exists(self, train_ctx: 'TrainTaskContext', *, epoch=None, step=None) -> bool:
        mode, value = self._resolve_eval_point(epoch=epoch, step=step)
        if mode == "epoch":
            return train_ctx.does_weight_exists(value)
        return train_ctx.does_step_weight_exists(value)

    def make_eval_task(
        self,
        eval_dataset_id,
        epoch=None,
        step=None,
        check_weight: bool = False,
        train_ctx: Optional['TrainTaskContext'] = None,
    ) -> Optional[GCHRecord]:
        """
            check_weight가 True이면 해당 epoch/step weight가 있을 때만 새 task를 생성한다.
            이미 task가 있으면 weight 여부와 관계없이 반환한다.
            train_ctx가 넘어오면 weight 체크에 재사용하고, 없으면 필요할 때 조회한다.
        """
        mode, value = self._resolve_eval_point(epoch=epoch, step=step)
        record = self.get_eval_task_record(
            eval_dataset_id,
            epoch=value if mode == "epoch" else None,
            step=value if mode == "step" else None,
        )
        if record is None:
            if check_weight:
                if train_ctx is None:
                    train_ctx = self._get_train_task_context()
                if train_ctx is None:
                    return None
                if not self._eval_checkpoint_exists(
                    train_ctx,
                    epoch=value if mode == "epoch" else None,
                    step=value if mode == "step" else None,
                ):
                    return None
            record_name = self.make_eval_record_name(
                eval_dataset_id,
                epoch=value if mode == "epoch" else None,
                step=value if mode == "step" else None,
            )
            record = self.task_db.create_record(record_name)
            eval_context = self.get_eval_task_context(record)
            eval_context.initialize(
                eval_dataset_id,
                epoch=value if mode == "epoch" else None,
                step=value if mode == "step" else None,
            )
    
        return record


    def make_all_eval_tasks(self, check_weight: bool = True)->None:
        metadata = self.metadata

        train_ctx = self._get_train_task_context() if check_weight else None
        print(metadata.get("eval_epochs", []))
        for dataset_id in metadata["eval_dataset_ids"]:
            for id in metadata.get("eval_epochs", []):
                self.make_eval_task(
                    dataset_id,
                    epoch=id,
                    check_weight=check_weight,
                    train_ctx=train_ctx,
                )
            for id in metadata.get("eval_steps", []):
                self.make_eval_task(
                    dataset_id,
                    step=id,
                    check_weight=check_weight,
                    train_ctx=train_ctx,
                )



    def get_eval_command(
        self,
        eval_dataset_id,
        epoch=None,
        step=None,
        *,
        check_result: bool = True,
        check_weight: bool = True,
        check_range: bool = True,
    ) -> Optional[str]:
        record = self.get_eval_task_record(eval_dataset_id, epoch=epoch, step=step)
        if record is None:
            return None
        context = self.get_eval_task_context(record)
        metadata = context.metadata
        if metadata['type'] != "test":
            return None
        task_epoch = metadata.get('epoch', None)
        task_step = metadata.get('step', None)
        if (task_epoch is None) == (task_step is None):
            raise ValueError(
                "Eval task metadata must contain exactly one of epoch or step."
            )
        if check_result and context.is_evaluated():
            return None

        if check_range and not self.is_eval_point_in_metadata(epoch=task_epoch, step=task_step):
            return None

        if check_weight:
            train_ctx = self._get_train_task_context()
            if train_ctx is None:
                return None
            if not self._eval_checkpoint_exists(train_ctx, epoch=task_epoch, step=task_step):
                return None
        return context.make_eval_command()

    

    def get_all_eval_command(
        self,
        check_result: bool = True,
        check_weight: bool = True,
        check_range: bool = True,
    ) -> List[str]:
        commands = []
        for id in self.task_db.ids:
            record = self.task_db.get_record(id)
            context = self.get_eval_task_context(record)
            metadata = context.metadata
            if metadata['type'] != "test":
                continue
            cmd = self.get_eval_command(
                metadata['dataset_id'],
                epoch=metadata.get('epoch'),
                step=metadata.get('step'),
                check_result=check_result,
                check_weight=check_weight,
                check_range=check_range,
            )
            if cmd is not None:
                commands.append(cmd)
        return commands


    @property
    def all_eval_checkpoints_exist(self) -> bool:
        train_ctx = self._get_train_task_context()
        if train_ctx is None:
            return False
        work_metadata = self.metadata
        for epoch in work_metadata.get("eval_epochs", []):
            if not self._eval_checkpoint_exists(train_ctx, epoch=epoch):
                return False
        for step in work_metadata.get("eval_steps", []):
            if not self._eval_checkpoint_exists(train_ctx, step=step):
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


    def make_eval_command(self)->str:
        return f"python /home/src/atomicstr/openocr/tools/eval_rec.py --work_id {self.work_record.id} --task_id {self.task_record.id}"




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

    def make_train_command(self)->str:
        return f"python /home/src/atomicstr/openocr/tools/train_rec.py --work_id {self.work_record.id} --task_id {self.task_record.id}"
    

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


    def make_train_command(self)->str:
        return f"python /home/src/atomicstr/openocr/tools/train_rec.py --work_id {self.work_record.id} --task_id {self.task_record.id}"
    
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