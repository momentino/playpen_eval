# Adding a new task in playpen_eval

There are two possible ways to upload new tasks to the playpen_eval benchmark suite. 
1. Through the [lm-eval-harness (our fork)](https://github.com/momentino/lm-evaluation-harness).
2. Outside of it, in our custom framework which we will call _playeval_framework_.

These are two evaluation frameworks which enable the creation of tasks in a unified format.
They can be found in the [_/frameworks_](https://github.com/momentino/playpen_eval/tree/main/frameworks) folder.

Why do we have two separate frameworks, if we wish to enable a unified interface for the evaluation?
It's because the _lm-eval-harness_ currently doesn't support certain kinds of benchmarks, and we do not want to modify it. We want to ensure compatibility with the original project.  
For this reason, all the benchmarks which are not supported by it will be implemented in _playpeval_framework_.

## Adding new tasks to lm-eval-harness
Follow the original documentation [here](https://github.com/momentino/lm-evaluation-harness/tree/main/docs).
## Adding new tasks to playeval_framework

This is what we have to figure out. However, we may implement new tasks within the _playeval_framework/tasks_ folder, starting from forking the repositories and them using **git submodule add <repo-url>**. This way, we still have a connection with the original project.  
We have to fork so we can also modify them.

## After implementing a task
You should define an entry in the [_/config/task_registry.yaml_](https://github.com/momentino/playpen_eval/blob/main/config/task_registry.yaml) file.
You don't have to add everything (many are for the correlation analysis), but some fields are fundamental. Let's look at an example:

 ```yaml
 entity_deduction_arena:  # Task name
      alias: "Entity Deduction Arena"  # Pretty name for the task
      functional: [ "theory_of_mind" ]  # the functional capabilities evaluated by the task 
      formal: [ ]   # formal capabilities evaluated by the task
      functional_groups: [ "social_emotional_cognition" ]  # functional macro-group
      task_type: [ "interactive" ]  # the type of task
      has_subtasks: False  # whether it has subtasks we want to consider separately
      main_task: True  # whether it is the main task (always true if it has no subtasks as well)
      group: entity_deduction_arena  # the group of tasks 
      backend: "playeval_framework"  # the type of backend
  ```
**The only fundamental fields for executing the benchmark are:**
- backend
- main_task
- has_subtasks


