import collections
import logging
from functools import partial
from typing import Mapping, Optional, Union

from lm_eval.api.group import ConfigurableGroup, GroupConfig
from lm_eval import utils
from lm_eval.tasks import TaskManager
from lm_eval_extended.api.task import ExtendedConfigurableTask
#from lm_eval.api.task import ConfigurableTask


GROUP_ONLY_KEYS = list(GroupConfig().to_dict().keys())

eval_logger = logging.getLogger(__name__)

class ExtendedTaskManager(TaskManager):
    def _load_individual_task_or_group(
            self,
            name_or_config: Optional[Union[str, dict]] = None,
            parent_name: Optional[str] = None,
            update_config: Optional[dict] = None,
    ) -> Mapping:
        def _load_task(config, task):
            if "include" in config:
                config = {
                    **utils.load_yaml_config(
                        yaml_path=None,
                        yaml_config={"include": config.pop("include")},
                        mode="full",
                    ),
                    **config,
                }
            if self._config_is_python_task(config):
                if self._class_has_config_in_constructor(config["class"]):
                    task_object = config["class"](config=config)
                else:
                    task_object = config["class"]()
                if isinstance(task_object, ExtendedConfigurableTask):
                    # very scuffed: set task name here. TODO: fixme?
                    task_object.config.task = task
            else:
                task_object = ExtendedConfigurableTask(config=config)

            return {task: task_object}

        def _get_group_and_subtask_from_config(config):
            group_name = ConfigurableGroup(config=config)
            subtask_list = []
            for task in group_name.config["task"]:
                if isinstance(task, str) and self._name_is_tag(task):
                    subtask_list.extend(self._get_tasklist(task))
                else:
                    subtask_list.append(task)
            return group_name, subtask_list

        def _process_group_config(config, update_config=None):
            if update_config is not None:
                config = {**config, **update_config}
            _update_config = {
                k: v for k, v in config.items() if k not in GROUP_ONLY_KEYS
            }
            if not bool(_update_config):
                _update_config = None

            group_config = {k: v for k, v in config.items() if k in GROUP_ONLY_KEYS}
            return group_config, _update_config

        if isinstance(name_or_config, str):
            if update_config is not None:
                # Process name_or_config as a dict instead
                name_or_config = {"task": name_or_config, **update_config}
            elif self._name_is_task(name_or_config) or self._name_is_python_task(
                    name_or_config
            ):
                task_config = self._get_config(name_or_config)
                return _load_task(task_config, task=name_or_config)
            else:
                subtask_list = self._get_tasklist(name_or_config)
                if subtask_list == -1:
                    group_config = self._get_config(name_or_config)
                    group_config, update_config = _process_group_config(group_config)
                    group_name, subtask_list = _get_group_and_subtask_from_config(
                        group_config
                    )
                else:
                    if self._name_is_tag(name_or_config):
                        fn = partial(
                            self._load_individual_task_or_group,
                            update_config=name_or_config
                            if isinstance(name_or_config, dict)
                            else None,
                        )
                        return dict(
                            collections.ChainMap(*map(fn, reversed(subtask_list)))
                        )
                    else:
                        group_name = ConfigurableGroup(
                            config={"group": name_or_config, "task": subtask_list}
                        )

        if isinstance(name_or_config, dict):
            if self._config_is_task(name_or_config):
                name = name_or_config.pop("task")
                if update_config is not None:
                    name_or_config = {**name_or_config, **update_config}
                # If the name is registered as a group
                if self._name_is_group(name):
                    group_config = self._get_config(name)

                    group_config, update_config = _process_group_config(
                        group_config, name_or_config
                    )
                    group_name, subtask_list = _get_group_and_subtask_from_config(
                        group_config
                    )
                elif self._name_is_tag(name):
                    subtask_list = self._get_tasklist(name)
                    fn = partial(
                        self._load_individual_task_or_group,
                        update_config=name_or_config,
                    )
                    return dict(collections.ChainMap(*map(fn, reversed(subtask_list))))
                else:
                    if self._name_is_registered(name):
                        base_task_config = self._get_config(name)

                        # Check if this is a duplicate.
                        if parent_name is not None:
                            num_duplicate = len(
                                list(
                                    filter(
                                        lambda x: x.startswith(name),
                                        self.task_group_map[parent_name],
                                    )
                                )
                            )
                            if num_duplicate > 0:
                                name = f"{name}-{num_duplicate}"
                            self.task_group_map[parent_name].append(name)

                        task_config = {
                            **base_task_config,
                            **name_or_config,
                        }
                    else:
                        task_config = name_or_config
                    return _load_task(task_config, task=name)
            else:
                group_config, update_config = _process_group_config(name_or_config)
                group_name, subtask_list = _get_group_and_subtask_from_config(
                    group_config
                )

        fn = partial(
            self._load_individual_task_or_group,
            parent_name=group_name,
            update_config=update_config,
        )
        return {
            group_name: dict(collections.ChainMap(*map(fn, reversed(subtask_list))))
        }