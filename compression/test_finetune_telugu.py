# test_finetune_telugu.py — end-to-end check of finetune_telugu_3b.py on a tiny random Llama (CPU, ~1 min)
#   python compression/test_finetune_telugu.py
import json, os, subprocess, sys, tempfile
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import finetune_telugu_3b as ft

torch.manual_seed(0)
tmp = tempfile.mkdtemp()
QA = [("భారతదేశ రాజధాని ఏది?", "భారతదేశ రాజధాని న్యూఢిల్లీ."),
      ("తెలంగాణ రాజధాని ఏది?", "తెలంగాణ రాజధాని హైదరాబాద్."),
      ("ఇంజిన్ ఆయిల్ ఎప్పుడు మార్చాలి?", "సాధారణంగా ప్రతి పది వేల కిలోమీటర్లకు ఒకసారి మార్చాలి."),
      ("టైర్ ప్రెషర్ ఎంత ఉండాలి?", "మీ బండి మాన్యువల్‌లో చెప్పిన ప్రెషర్ ఉంచండి.")] * 10

# 1. tiny byte-level BPE tokenizer with a Llama-3 style chat template, saved like a HF model folder
specials = ["<|begin_of_text|>", "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"]
bpe = Tokenizer(models.BPE())
bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
bpe.decoder = decoders.ByteLevel()
corpus = [q + " " + a for q, a in QA] + [ft.SYSTEM_PROMPT]
bpe.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=specials,
                                                    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tok = PreTrainedTokenizerFast(tokenizer_object=bpe, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>")
tok.chat_template = ("{{ bos_token }}{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n"
                     "{{ m['content'] }}<|eot_id|>{% endfor %}"
                     "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}")
base_dir = os.path.join(tmp, "base")
cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
                  bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
LlamaForCausalLM(cfg).save_pretrained(base_dir); tok.save_pretrained(base_dir)
tok.pad_token = tok.eos_token

# 2. data readers: jsonl, csv, and {"messages": [...]}
jsonl = os.path.join(tmp, "qa.jsonl"); csv_path = os.path.join(tmp, "qa.csv")
with open(jsonl, "w", encoding="utf-8") as f:
    for q, a in QA[:20]:
        f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
    f.write(json.dumps({"messages": [{"role": "user", "content": QA[0][0]}, {"role": "assistant", "content": QA[0][1]},
                                     {"role": "user", "content": QA[1][0]}, {"role": "assistant", "content": QA[1][1]}]},
                       ensure_ascii=False) + "\n")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("instruction,output\n" + "".join(f"{q},{a}\n" for q, a in QA[20:]))
convs = ft.read_rows(jsonl) + ft.read_rows(csv_path)
assert len(convs) == 41 and len(convs[20]) == 4
print("ok  read .jsonl (question/answer + messages) and .csv (instruction/output)")

# 3. only assistant tokens are trained, every assistant turn is trained
ids, labels = ft.encode_conversation(convs[20], tok, ft.SYSTEM_PROMPT, 4096)
trained = tok.decode([t for t, l in zip(ids, labels) if l != ft.IGNORE])
assert trained == QA[0][1] + "<|eot_id|>" + QA[1][1] + "<|eot_id|>", trained
assert ft.SYSTEM_PROMPT.splitlines()[0] in tok.decode(ids)
assert ft.telugu_ratio("భారతదేశ రాజధాని") == 1.0 and ft.telugu_ratio("hello") == 0.0
print("ok  loss mask covers exactly the assistant answers (multi-turn)")

# 4. full CLI: train (with resume), merge, eval
out = os.path.join(tmp, "out")
script = os.path.join(HERE, "finetune_telugu_3b.py")
common = ["--base", base_dir, "--out", out]
run = lambda *a: subprocess.run([sys.executable, script, *a, *common], check=True, capture_output=True, text=True).stdout
log = run("train", "--qa", jsonl, csv_path, "--steps", "4", "--batch-size", "4", "--grad-accum", "1",
          "--lr", "5e-3", "--warmup", "1", "--log-every", "1", "--save-every", "2", "--lora-r", "8")
assert "step     4/4" in log and os.path.exists(os.path.join(out, "adapter", "adapter_config.json")), log
log = run("train", "--qa", jsonl, csv_path, "--steps", "6", "--batch-size", "4", "--grad-accum", "1",
          "--lr", "5e-3", "--warmup", "1", "--log-every", "1", "--save-every", "2", "--lora-r", "8")
assert "resumed from step 4" in log and "step     6/6" in log, log
print("ok  train stage + resume")

log = run("merge")
merged = LlamaForCausalLM.from_pretrained(os.path.join(out, "merged"))
base = LlamaForCausalLM.from_pretrained(base_dir)
assert not torch.equal(merged.model.layers[0].self_attn.q_proj.weight, base.model.layers[0].self_attn.q_proj.weight)
assert not any("lora" in k for k in merged.state_dict())
print("ok  merge stage -> plain HF model with LoRA folded in")

log = run("eval", "--qa", jsonl, csv_path, "--n-samples", "2")
assert "base        held-out answer loss" in log and "fine-tuned  held-out answer loss" in log, log
print("ok  eval stage")
print(log.strip().splitlines()[-1][:80])
print("ALL TESTS PASSED")
