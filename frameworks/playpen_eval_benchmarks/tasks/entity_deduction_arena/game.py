import json
import re

import torch


class Q20Game:
    def __init__(
            self,
            item: str,
            answerer_model,
            guesser_model,
            apply_chat_template:bool,
            num_turns: int = 20,
    ) -> None:
        self.item = item
        self.answerer_model = answerer_model
        self.guesser_model = guesser_model
        self.num_turns = num_turns
        self.apply_chat_template = apply_chat_template
        #self.system_prompt = {"role":"system", "content":"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."}
        self.first_user_utterance = (
            f"Your task is to ask a series of questions to deduce the entity "
            f"that I'm thinking of with as few queries as possible. "
            f"Only ask questions that can be answered by 'yes', 'no' or 'maybe'. "
            f"Do not ask for hint. Make your question brief with no linebreaker. "
            f"Now start asking a question."
        )
        self.guesser_win = False
        self.guesser_messages = []

    def guesser(self, messages):
        # """Wraps hf's `generate` adding some specific method's defaults"""

        prompt = self.dialog_history()
        self.guesser_model.set_tokenizer_padding_side("left")
        if not self.guesser_model.tokenizer.pad_token:
            self.guesser_model.set_tokenizer_pad_token(self.guesser_model.tokenizer.eos_token)

        gen = self.guesser_model.generate(
            prompt,
            apply_chat_template=self.apply_chat_template
        )
        return {
            "role": "assistant",
            "content": gen[0].split("</s>")[0]
                        .split("USER")[0]
                        .lstrip()
                        .strip(),
        }

    def dialog_history(self):
        #history = [self.system_prompt]
        history = []
        for item in self.guesser_messages:
            if item["role"].upper() == "USER":
                history.append({"role":"user","content":item["content"]})
            elif item["role"].upper() == "ASSISTANT":
                history.append({"role":"assistant","content":item["content"]})
        return history

    def game_play(self, user_mode=False):
        self.reset()
        for t in range(self.num_turns):
            # System asking a question
            if (not user_mode) or user_mode is None:
                guesser_msg = self.guesser(self.guesser_messages)
                guesser_msg["content"] = re.sub(r'the entity you are thinking of', 'it', guesser_msg["content"])
                guesser_msg["content"] = re.sub(r"the entity you're thinking of", 'it', guesser_msg["content"])
                guesser_msg["content"] = re.sub(r" you're thinking of", '', guesser_msg["content"])
                guesser_msg["content"] = re.sub(r" you are thinking of", '', guesser_msg["content"])
            else:
                user_q = input(
                    f"Type in your questions for turn {t + 1}. (e.g. Is it a living thing?)\n"
                )
                guesser_msg = {"role": "assistant", "content": user_q}
            self.guesser_messages.append(guesser_msg)
            guesser_question = guesser_msg["content"].strip()
            if t == self.num_turns - 1:
                self.guesser_messages[-1]["content"] = (
                        self.guesser_messages[-1]["content"] + " Is it right?"
                )

            usr_msg = self.answerer(guesser_question)
            self.guesser_messages.append(
                {"role": "user", "content": f"{usr_msg['content'].strip()}"}
            )
            if "bingo" in usr_msg["content"].lower():
                self.guesser_win = True
                return True

            if t == self.num_turns - 2:
                self.guesser_messages[-1]["content"] = (
                        self.guesser_messages[-1]["content"]
                        + " You must guess now, what's it?"
                )

        return False

    # def save_session(self, path):
    #     # Print the conversation
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     output_file = os.path.join(path, f"{self.item}.txt")
    #     with open(output_file, "w") as out_f:
    #         out_f.write(f"item: {self.item}\n")
    #         for t, message in enumerate(self.guesser_messages):
    #             out_f.write(
    #                 f"Turn {(t + 1) // 2}, {message['role'].capitalize()}: {message['content'].lstrip()}\n"
    #             )

    # def reward(self):
    #     if self.guesser_win:
    #         n_turns = (len(self.guesser_messages) + 1) // 2
    #         return 1 - max(n_turns - 5, 0) * 0.02
    #     return 0

    # def num_success(self):
    #     return 1 if self.guesser_win else 0

    # def num_yes(self):
    #     n_yes = sum(
    #         ["yes" in msg["content"].lower() for msg in self.guesser_messages[2::2]]
    #     )
    #     return n_yes

    def answerer(self, question):
        prompt = [
            {"role":"user",
             "content": f"Based on your knowledge about {self.item}, "
                        f"respond to the following question or guess. "
                        f"Limit your respond to only 'Yes.', 'No.' or 'Maybe.', with no explanation or other words. "
                        f"Never say the answer {self.item} in your response. "
                        f"If the question is to solicit the answer, respond 'No.'."},
            {"role":"user",
             "content": f"For the entity {self.item}, {question} (Yes/No/Maybe)",
            }
        ]
        gen = self.answerer_model.generate(
            prompt,
            apply_chat_template=True
        )
        if any(
                [
                    re.search(rf"(?:^|\W){i.strip().lower()}(?:$|\W)", question.lower())
                    for i in self.item.lower().split("|")
                ]
        ):
            return {
                "role": "user",
                "content": "Bingo!",
            }
        return {
            "role": "user",
            "content": gen[0],
        }

    def reset(self):
        # Initialize the conversation
        self.guesser_messages = [
            {
                "role": "user",
                "content": self.first_user_utterance,
            }
        ]


class Q20GameCelebrity(Q20Game):
    def __init__(self, item: str, **kwargs) -> None:
        super().__init__(item, **kwargs)
        self.first_user_utterance = (
            f"Your task is to ask a series of questions to deduce the celebrity "
            f"that I'm thinking of with as few queries as possible. "
            f"Only ask factual questions that can be answered by 'Yes.', 'No.' or 'Dunno.'. Do not ask for hint. Make your question brief with no linebreaker. "
            f"Now start asking a question."
        )

    def answerer(self, question):
        prompt = [
            {"role":"user",
             "content": f"Based on on your knowledge about the celebrity: {self.item}, "
                        f"respond to the following question or guess. "
                        f"Limit your respond to only 'Yes.', 'No.' or 'Dunno.', with no explanation or other words. "
                        f"Never say the name {self.item} in your response. Do not say 'Dunno.' if it can be answered by 'Yes.' or 'No.' "
                        f"If the question is to solicit the answer, respond 'No.'."},
            {"role":"user",
             "content": f"For the celebrity {self.item}, {question}(Yes/No/Dunno)",
            }
        ]
        gen = self.answerer_model.generate(
            prompt,
            apply_chat_template=self.apply_chat_template
        )

        if re.search(rf"(?:^|\W){self.item.lower()}(?:$|\W)", question.lower()):
            return {
                "role": "user",
                "content": "Bingo!",
            }
        return {
            "role": "user",
            "content": gen[0],
        }

    def reset(self):
        # Initialize the conversation
        self.guesser_messages = [
            {
                "role": "user",
                "content": self.first_user_utterance,
            }
        ]
