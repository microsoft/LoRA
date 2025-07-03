LoRA with Hugging Face Models: A Beginner-Friendly Tutorial
===========================================================

Introduction
------------
This tutorial will guide you through integrating LoRA (Low-Rank Adaptation) with Hugging Face models for efficient fine-tuning. LoRA reduces the number of trainable parameters, making it ideal for adapting large language models to specific tasks.

Prerequisites
-------------
- Python 3.8 or later
- PyTorch installed (`pip install torch`)
- Hugging Face Transformers library (`pip install transformers`)
- LoRA library (`pip install loralib`)

Step 1: Install Dependencies
----------------------------
Run the following commands to install the required libraries:

.. code-block:: bash

    pip install torch transformers loralib

Step 2: Load a Pretrained Model
-------------------------------
We'll use the Hugging Face Transformers library to load a pretrained model. For this example, we'll use `google/pegasus-xsum` for summarization.

.. code-block:: python

    from transformers import PegasusForConditionalGeneration, PegasusTokenizer

    model_name = "google/pegasus-xsum"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)

Step 3: Integrate LoRA
----------------------
Replace the model's layers with LoRA layers to enable low-rank adaptation.

.. code-block:: python

    import loralib as lora

    # Replace the encoder and decoder layers with LoRA layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, lora.Linear(module.in_features, module.out_features, r=16))

Step 4: Fine-Tune the Model
---------------------------
Mark only LoRA parameters as trainable and fine-tune the model on your dataset.

.. code-block:: python

    lora.mark_only_lora_as_trainable(model)

    # Example training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for batch in dataloader:
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
        labels = tokenizer(batch["summary"], return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs, labels=labels["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

Step 5: Save and Load LoRA Checkpoints
--------------------------------------
Save only the LoRA parameters to reduce storage requirements.

.. code-block:: python

    torch.save(lora.lora_state_dict(model), "lora_checkpoint.pt")

To load the checkpoint:

.. code-block:: python

    model.load_state_dict(torch.load("lora_checkpoint.pt"), strict=False)

Step 6: Evaluate the Model
--------------------------
Use the fine-tuned model for inference.

.. code-block:: python

    src_text = [
        """PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires."""
    ]
    inputs = tokenizer(src_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

Conclusion
----------
Congratulations! You've successfully integrated LoRA with a Hugging Face model. This approach allows efficient fine-tuning with minimal storage requirements.

---

### Step 3: Add the Tutorial to the Documentation Index
Update the `/examples/NLU/docs/source/index.rst` file to include the new tutorial:

```restructuredtext
// filepath: [index.rst](http://_vscodecontentref_/0)

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   lora_huggingface_tutorial