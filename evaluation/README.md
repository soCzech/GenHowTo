# How to replicate our evaluation?

This is a step-by-step guide on how to replicate our evaluation.
Note, the whole process of evaluation is done for the final states and actions separately.

1. **Data**
    - Our evaluation is based on the [ChangeIt dataset](https://github.com/soCzech/ChangeIt).
      For the purpose of the evaluation, the held-out categories are: apple, avocado, bacon, ball, beer.
    - The held-out image pairs and test image pairs are available in the `data` directory.
      - `VID_ID` is YouTube video ID.
      - `DS_CAT` is the ChangeIt dataset category ID of the video.
      - `DS_CAT_NAME` is the ChangeIt dataset category name of the video.
      - `FRAME_S1` is the time in seconds of a frame used as the **input initial state image**.
      - `FRAME_S2` is the time in seconds of a frame used as the **target final state image**.
      - `FRAME_AC` is the time in seconds of a frame used as the **target action image**.
      - `SCORE` is the score returned by [the unsupervised model](https://github.com/soCzech/MultiTaskObjectStates) for the video.
      - `FRAME_S2_PROMPT` or `FRAME_AC_PROMPT` is the **input text prompt** for the target final state or action image respectively.
      - `FILENAME` name of a file containing the input and target images (the images are not provided due to copyright issues).
    - Due to copyright issues, we do not publicly provide the original train and test images used for training and evaluation.
      We can provide the data upon request sent to *tomas.soucek at cvut dot cz* specifiing your name and affiliation.
      Please use your institutional email (i.e. not gmail, etc.).
2. **Generate images**
    - Use the provided `genhowto.py` code to generate images of the held-out set.
      I.e., use the image `FRAME_S1` and the prompt `FRAME_(S2|AC)_PROMPT` from `data/(states|actions).heldout.txt` to generate target class images.
3. **Form train and test datasets for classification**
    - Real images `FRAME_S1` from `data/(states|actions).test.txt` form the test set classes `0`, `1`, `2`, `3`, `4`.
    - Real images `FRAME_(S2|AC)` from `data/(states|actions).test.txt` form the test set classes `5`, `6`, `7`, `8`, `9`.
    - Real images `FRAME_S1` from `data/(states|actions).heldout.txt` form the train set classes `0`, `1`, `2`, `3`, `4`.
    - The generated images form the train set classes `5`, `6`, `7`, `8`, `9`.
4. **Extract image features**
   - Use CLIP `ViT-L/14` from the official [CLIP repository](https://github.com/openai/CLIP) to extract 
     classifier-token features before the final projection layer for all images in the train and test set.
5. **Train and evaluate the classifier**
   - Use the extracted features to train a linear classifier.
   ```python
   reg = sklearn.linear_model.LogisticRegression(C=C, max_iter=max_iter).fit(X, y)
   accuracy = reg.score(X_test, y_test)
   ```
   - We used 10% of the train set as a validation set to select the best hyperparameters `C` and `max_iter`.
     - We considered `C` of `1e-07`, `5e-07`, `1e-06`, `1e-05`, and `1e-04`
     - We considered `max_iter` of `25`, `100`, and `250`

