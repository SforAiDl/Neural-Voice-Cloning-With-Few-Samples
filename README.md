**Status**: Archive (code is provided as-is, no updates expected)

# Neural-Voice-Cloning-with-Few-Samples


We are trying to clone voices for speakers which is content independent. This means that we have to encapture the identity of the speaker rather than the content they speak. We try to do this by making a speaker embedding space for different speakers.

The speaker embeddings try to represent the identity of the speaker(various aspects of the voice like pitch, accent, etc of the speaker), you can consider this as the voice fingerprint of the speaker.


We are right now referring to the following paper for our Implementation:-

- ["Neural Voice Cloning with Few Samples"](https://arxiv.org/pdf/1802.06006) by Baidu


### Status

The architecture for the Multi-Speaker Generative and Speaker Encoder Model have been built.

Multi-Speaker Generative model has been trained for speaker adaptation for 84 speakers using VCTK-dataset has been completed on NVIDIA - V100 GPU for 190000 epochs.


## Speaker Adapatation

VCTK dataset was split for training and testing: 84 speakers are used for training
the multi-speaker model, 8 speakers for validation, and 16 speakers for cloning.

#### Training for Speaker Adapatation

The following will train the model on the first 84 speakers in the dataset.

```
python speaker_adaptation.py --data-root=<path_of_vctk_dataset> --checkpoint-dir=<path> --checkpoint-interval=<int>
```

This can take upto 20 hours using a GPU.

To adapt the model to a particular speaker after the initial training

```
python speaker_adaptation.py --data-root=<path_of_vctk_dataset> --restore-parts=<path_of_checkpoint> --checkpoint-dir=<path> --checkpoint-interval=<int>

```

This will take on an average of 10 to 20 minutes.


#### Some Cloned Voices


So far some of the coned voices we have got using speaker adaptation [LINK](https://sforaidl.github.io/Neural-Voice-Cloning-With-Few-Samples/)









# Acknowledgements

- The implementation of Multi-Speaker Generative model was inspired from https://github.com/r9y9/deepvoice3_pytorch

- [Neural Voice Cloning with Few Samples](https://arxiv.org/pdf/1802.06006)


# Cite

If you find the code in the repository useful, please cite it using:

```
@misc{chitlangia2021voicecloning,
  author = {Chitlangia, Sharad and Rastogi, Mehul and Ganguly, Rijul},
  title = {An Open Source Implementation of Neural Voice Cloning With Few Samples},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {Available at \url{https://github.com/SforAiDl/Neural-Voice-Cloning-With-Few-Samples/} or \url{https://github.com/Sharad24/Neural-Voice-Cloning-With-Few-Samples/}},
}
```
