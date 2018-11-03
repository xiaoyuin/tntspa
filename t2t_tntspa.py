# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import os

@registry.register_problem
class TranslateEnsparql(translate.TranslateProblem):
    """Problem spec for English-SPARQL translation."""

    @property
    def vocab_type(self):
        return text_problems.VocabType.TOKEN

    @property
    def oov_token(self):
        return "<unk>"

    @property
    def is_generate_per_split(self):
        return True

    @property
    def source_vocab_filename(self):
        return "vocab.en"

    @property
    def target_vocab_filename(self):
        return "vocab.sparql"

    @property
    def vocab_filename(self):
        return "vocab.shared"

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data 10% test data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        },{
            "split": problem.DatasetSplit.TEST,
            "shards": 1,
        }]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del tmp_dir

        prefix = "train" if dataset_split == problem.DatasetSplit.TRAIN else ("dev" if dataset_split == problem.DatasetSplit.EVAL else "test")

        en_dir = os.path.join(data_dir, "{}.en".format(prefix))
        sparql_dir = os.path.join(data_dir, "{}.sparql".format(prefix))
       
        return text_problems.text2text_txt_iterator(en_dir, sparql_dir)

    # def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        
    #     source_vocab_dir = os.path.join(data_dir, self.source_vocab_filename)
    #     target_vocab_dir = os.path.join(data_dir, self.target_vocab_filename)
    #     source_txt_encoder = text_encoder.TokenTextEncoder(source_vocab_dir, replace_oov=self.oov_token)
    #     target_txt_encoder = text_encoder.TokenTextEncoder(target_vocab_dir, replace_oov=self.oov_token)

    #     return text_problems.text2text_generate_encoded(
    #         self.generate_samples(data_dir, tmp_dir, dataset_split),
    #         source_txt_encoder, target_txt_encoder
    #         )