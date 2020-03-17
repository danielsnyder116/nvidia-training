from __future__ import absolute_import, division, print_function
import tensorflow as tf

from open_seq2seq2.models import Text2Text
from open_seq2seq2.encoders import BidirectionalRNNEncoderWithEmbedding
from open_seq2seq2.decoders import RNNDecoderWithAttention, BeamSearchRNNDecoderWithAttention
from open_seq2seq2.data.text2text.text2text import ParallelTextDataLayer
from open_seq2seq2.losses import BasicSequenceLoss
from open_seq2seq2.data.text2text.text2text import SpecialTextTokens
from open_seq2seq2.optimizers.lr_policies import fixed_lr

base_model = Text2Text

base_params = {
  "use_horovod": False,
  "num_gpus": 1,
  "num_epochs": 40,
  "print_loss_steps": 500,
  "print_samples_steps": 500,
  "batch_size_per_gpu": 128,
  "save_checkpoint_steps": 2500,
  "logdir": "dates_log",
  "optimizer": "RMSProp",
  "optimizer_params": {},
  "lr_policy": fixed_lr,
  "lr_policy_params": {
    "learning_rate": 0.001,
  },
  "dtype": tf.float32,
  
  "encoder": BidirectionalRNNEncoderWithEmbedding,
  "encoder_params": {
    "initializer": tf.glorot_uniform_initializer,
    "core_cell": tf.nn.rnn_cell.LSTMCell,
    "core_cell_params": {
        "num_units": 32,
        "forget_bias": 1.0,
    },
    "encoder_layers": 1,
    "encoder_dp_input_keep_prob": 0.8,
    "encoder_dp_output_keep_prob": 1.0,
    "src_emb_size": 64,
    "encoder_use_skip_connections": False
  },

  "decoder": RNNDecoderWithAttention,
  "decoder_params": {
    "initializer": tf.glorot_uniform_initializer,
    "core_cell": tf.nn.rnn_cell.LSTMCell,
     "core_cell_params": {
        "num_units": 32,
        "forget_bias": 1.0,
    },
    "decoder_layers": 1,
    "decoder_dp_input_keep_prob": 0.8,
    "decoder_dp_output_keep_prob": 1.0,
    "tgt_emb_size": 64,
    "attention_type": "bahdanau",
    "attention_layer_size": 128,
    "GO_SYMBOL": SpecialTextTokens.S_ID.value,
    "END_SYMBOL": SpecialTextTokens.EOS_ID.value,
    "PAD_SYMBOL": SpecialTextTokens.PAD_ID.value,
    "decoder_use_skip_connections": False,
  },

  "loss": BasicSequenceLoss,
  "loss_params": {
    "offset_target_by_one": True,
    "average_across_timestep": False,
    "do_mask": True
  }
}

train_params = {
  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "src_vocab_file": "dates/source_vocab.txt",
    "tgt_vocab_file": "dates/target_vocab.txt",
    "source_file": "dates/source_train.txt",
    "target_file": "dates/target_train.txt",
    "delimiter": "|",
    "shuffle": True,
    "repeat": True,
    "map_parallel_calls": 16,
    "prefetch_buffer_size": 2,
    "max_length": 60,   
  },
}

infer_params = {
  "batch_size_per_gpu": 1,
  "decoder": BeamSearchRNNDecoderWithAttention,
  "decoder_params": {
    "beam_width": 1,
    "length_penalty": 0.0,
    "initializer": tf.glorot_uniform_initializer,
    "core_cell": tf.nn.rnn_cell.LSTMCell,
     "core_cell_params": {
        "num_units": 32,
        "forget_bias": 1.0,
    },
    "decoder_layers": 1,
    "decoder_dp_input_keep_prob": 0.8,
    "decoder_dp_output_keep_prob": 1.0,
    "tgt_emb_size": 64,
    "attention_type": "bahdanau",
    "attention_layer_size": 128,
    "GO_SYMBOL": SpecialTextTokens.S_ID.value,
    "END_SYMBOL": SpecialTextTokens.EOS_ID.value,
    "PAD_SYMBOL": SpecialTextTokens.PAD_ID.value,
  },
  "data_layer": ParallelTextDataLayer,
  "data_layer_params": {
    "src_vocab_file": "dates/source_vocab.txt",
    "tgt_vocab_file": "dates/target_vocab.txt",
    "source_file": "dates/source_test.txt",
    # Target_file is wrong - this is ignored during inference
    "target_file": "dates/source_test.txt",
    "delimiter": "|",
    "shuffle": False,
    "repeat": False,
    "map_parallel_calls": 16,
    "prefetch_buffer_size": 2,
    "max_length": 60,
  },
}
  

 
