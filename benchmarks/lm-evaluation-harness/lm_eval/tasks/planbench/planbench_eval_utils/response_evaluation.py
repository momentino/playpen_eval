""" Adapted from the original repository for this benchmark:
https://github.com/karthikv792/LLMs-Planning/tree/main/plan-bench """
import os
import json
from pathlib import Path
from lm_eval.tasks.planbench.planbench_eval_utils.text_to_pddl import text_to_plan, text_to_state
from lm_eval.tasks.planbench.planbench_eval_utils.Executor import Executor
from tarski.io import PDDLReader
from lm_eval.tasks.planbench.planbench_eval_utils.model_parser.writer_new import ModelWriter

current_folder = Path(__file__).parent.resolve()

def validate_plan(domain, instance, plan_file):
    val_path = current_folder / "VAL"
    cmd = f"{val_path}/validate {domain} {instance} {plan_file}"
    response = os.popen(cmd).read()
    if 'Problem in domain' in response:
        raise Exception('Problem in domain: Check PDDL Writer')
    return True if "Plan valid" in response else False

class ResponseEvaluator:
    def __init__(self, config_file):
        self.data = self.read_config(config_file)
        self.instance_dir = self.data['instance_dir']
        self.domain_pddl = current_folder /"instances"/self.data["domain_file"]
        self.llm_plan_file = 'llm_plan'
        self._set_task_params()

    def read_config(self, config_file):
        with open(config_file, 'r') as file:
            return json.load(file)

    def _set_task_params(self, instance_dir=None):
        if instance_dir is None:
            instance_dir = self.instance_dir
        self.instance_folder = current_folder /'instances'/instance_dir
        self.instance = current_folder / 'instances'/instance_dir/self.data["instances_template"]
        self.n_files = min(self.data['n_instances'], len(os.listdir(self.instance_folder)))

        self.i_start = self.data['start']
        self.i_end = self.data['end']

    def get_problem(self, instance, domain):
        reader = PDDLReader(raise_on_error=True)
        reader.parse_domain(domain)
        return reader.parse_instance(instance)

    def get_executor(self, instance, domain, ground=False):
        plan_executor = Executor(domain, instance, ground=ground)
        return plan_executor

    def write_new_instance(self, new_model):
        writer = ModelWriter(new_model)
        writer.write_files('pr-new-domain.pddl', 'pr-new-problem.pddl')

    def evaluate_plan(self, task_name, doc, llm_response):
        if 'plan_generalization' in task_name:
            self._set_task_params(instance_dir=self.data['generalized_instance_dir'])
        id = doc["instance_id"]
        cur_instance = str(self.instance).format(id)
        problem = self.get_problem(cur_instance, self.domain_pddl)
        plan_executor = self.get_executor(cur_instance, self.domain_pddl)
        try:
            llm_plan, _ = text_to_plan(llm_response, problem.actions, self.llm_plan_file, self.data)
            if 'new_instance' not in doc.keys():
                correct = int(validate_plan(self.domain_pddl, cur_instance, self.llm_plan_file))
            else:
                self.write_new_instance(doc['new_instance'])
                correct = int(validate_plan('pr-new-domain.pddl', 'pr-new-problem.pddl', self.llm_plan_file))
                # remove new_instance key from instance_dict
                del doc['new_instance']
            if 'optimality' in task_name:
                if correct:
                    plan_list = [len(pl) > 0 for pl in llm_plan.split('\n')]
                    actual_cost_llm = sum(plan_list)
                    if actual_cost_llm == plan_executor.cost:
                        correct = True
                    else:
                        correct = False
        except:
            # Plan extraction failed
            correct = int(False)
            print(f"Warning: Plan extraction failed for plan {id}")

        """try:
            llm_plan, _ = text_to_plan(llm_response, problem.actions, self.llm_plan_file, self.data)
            if 'new_instance' not in doc.keys():
                correct = int(validate_plan(self.domain_pddl, cur_instance, self.llm_plan_file))
            else:
                self.write_new_instance(doc['new_instance'])
                correct = int(validate_plan('pr-new-domain.pddl', 'pr-new-problem.pddl', self.llm_plan_file))
                # remove new_instance key from instance_dict
                del doc['new_instance']
            if 'optimality' in task_name:
                if correct:
                    plan_list = [len(pl) > 0 for pl in llm_plan.split('\n')]
                    actual_cost_llm = sum(plan_list)
                    if actual_cost_llm == plan_executor.cost:
                        correct = True
                    else:
                        correct = False
        except:
            # Plan extraction failed
            correct = int(False)
            print(f"Warning: Plan extraction failed for plan {id}")"""
        return correct

    def evaluate_state(self, doc, llm_response):

        ground_state = doc["ground_truth_plan"]
        llm_state = text_to_state(llm_response, self.data)

        if sorted(ground_state) == sorted(llm_state):
            correct = True
        else:
            correct = False
        return correct

    def parse_output(self, action_set, output):
        output_dict = {}
        goal_cond = False
        precond_act = False
        precond_act_flag = False
        precond_pred = False
        for line in output.split('\n'):
            if '[STATEMENT]' in line:
                break
            if line.strip() == "":
                continue
            if goal_cond:
                output_dict['unmet_goal'] = text_to_state(line.strip(), self.data)
                goal_cond = False
                continue
            if precond_act:
                _, action = text_to_plan(line.strip(), action_set, self.llm_plan_file, self.data)
                output_dict['unmet_precondition']['action'] = action
                precond_act = False
                precond_act_flag = True
                continue
            if precond_act_flag and precond_pred:
                # print(line.strip(), text_to_state(line.strip(), self.data))
                output_dict['unmet_precondition']['predicate'] = text_to_state(line.strip(), self.data)
                precond_pred = False
                precond_act_flag = False

            if 'plan is valid' in line:
                if 'valid' not in output_dict:
                    output_dict['valid'] = True
                break
            elif 'plan is invalid' in line:
                output_dict['valid'] = False
            if 'unmet goal' in line and 'unmet precondition' in line:
                break
            if 'unmet goal' in line:
                output_dict['unmet_goal'] = ''
                goal_cond = True
            elif 'unmet precondition' in line:
                if 'action' in line:
                    output_dict['unmet_precondition'] = {}
                    precond_act = True
                else:
                    precond_pred = True
            elif 'Unmet precondition:' in line:
                if precond_act_flag:
                    output_dict['unmet_precondition']['predicate'] = text_to_state(line.strip(), self.data)
                    precond_act_flag = False

        return output_dict

    def evaluate_verification(self, doc, llm_response):
        id = doc["instance_id"]
        cur_instance = str(self.instance).format(id)
        problem = self.get_problem(cur_instance, self.domain_pddl)
        ground_truth_response = doc["ground_truth_plan"]
        parsed_llm_response = self.parse_output(problem.actions, llm_response)
        parsed_ground_truth_response = self.parse_output(problem.actions, ground_truth_response)
        correct_binary = False
        correct_w_type = False
        correct_w_expl = False
        try:
            if parsed_llm_response['valid'] == parsed_ground_truth_response['valid']:
                correct_binary = True
                if not parsed_llm_response['valid']:
                    # print(sorted(list(parsed_llm_response.keys())), sorted(list(parsed_ground_truth_response.keys())), sorted(list(parsed_llm_response.keys())) == sorted(list(parsed_ground_truth_response.keys())))
                    if sorted(list(parsed_llm_response.keys())) == sorted(
                            list(parsed_ground_truth_response.keys())):
                        correct_w_type = True
                        if 'unmet_goal' in parsed_llm_response:
                            # if parsed_ground_truth_response['unmet_goal'] == parsed_llm_response['unmet_goal']:
                            if any([llm_pred in parsed_ground_truth_response['unmet_goal'] for llm_pred in
                                    parsed_llm_response['unmet_goal']]):
                                correct_w_expl = True
                        if 'unmet_precondition' in parsed_llm_response:
                            try:
                                if parsed_llm_response['unmet_precondition']['action'] == \
                                        parsed_ground_truth_response['unmet_precondition']['action']:
                                    if any([llm_pred in parsed_ground_truth_response['unmet_precondition'][
                                        'predicate'] for llm_pred in
                                            parsed_llm_response['unmet_precondition']['predicate']]):
                                        correct_w_expl = True

                            except KeyError:
                                print(f"For Instance {id}")
                                print(parsed_llm_response)
                                print(parsed_ground_truth_response)
                            # raise KeyError

            else:
                correct_w_type = True
                correct_w_expl = True
        except KeyError:
            print("Plan verification failed")
        return correct_binary and correct_w_type and correct_w_expl