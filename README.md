# Self-Supervised Learning of Visual Robot Localization Using Prediction of LEDs States as a Pretext Task

*Mirko Nava, Nicola Armas, Antonio Paolillo, Jerome Guzzi, Luca Maria Gambardella, and Alessandro Giusti*

Dalle Molle Institute for Artificial Intelligence, USI-SUPSI, Lugano (Switzerland)

### Abstract

We propose a novel self-supervised approach to learn CNNs that perform visual localization of a robot in an image using very small labeled training datasets. Self-supervision is obtained by jointly learning a pretext task, i.e., predicting the state of the LEDs of the target robot.
This pretext task is compelling because: a) it indirectly forces the model to learn to locate the target robot in the image in order to determine its LED states; b) it can be trained on large datasets collected in any environment with no external supervision or tracking infrastructure.
We instantiate the general approach to a concrete task: visual relative localization of nano-quadrotors.
Experimental results on a challenging dataset show that the approach is very effective; compared to a baseline that does not use the proposed pretext task, it reduces the mean absolute localization error by as much as 78% (43 to 9 pixels on *x*; 28 to 6 pixels on *y*).

<!--
![Sound as Pretext](https://github.com/idsia-robotics/Sound-as-Pretext/blob/main/code/data/out/Intro.png)
Figure 1: *Given an image from the ground robot camera, the model estimates the relative position of the drone; this is the **end task**, learned by minimizing a regression end loss on few training frames for which the true relative position is known.
We show that simultaneously learning to predict audio features (**pretext task**), which are known in all training frames, yields dramatic performance improvements for the end task.*


![Regression Performance on the testing set](https://github.com/idsia-robotics/Sound-as-Pretext/blob/main/code/data/out/results.png)
Figure 2: *End Task Regression Performance on the testing set.
On the left side, we compare ground truth (x axis) and predictions (y axis) for different models (columns) and variables (rows).
On the right, predictions on 30s of the testing set.
Between seconds 17 and 20 the drone exits of the camera FOV, causing all models to temporarily fail.*

The PDF of the article is available in Open Access [here]( https://doi.org/10.1109/LRA.2022.3143565).

### Bibtex will be displayed here later

```properties
@article{nava2022learning,
  author={M. {Nava} and A. {Paolillo} and J. {Guzzi} and L. M. {Gambardella} and A. {Giusti}},
  journal={IEEE Robotics and Automation Letters}, 
  title={Learning Visual Localization of a Quadrotor Using its Noise as Self-Supervision}, 
  year={2022},
  volume={7},
  number={2},
  pages={2218-2225},
  doi={10.1109/LRA.2022.3143565}
}
```

### Video

[![Learning Visual Object Localization from Few Labeled Examples using Sound Prediction as a Pretext Task](https://github.com/idsia-robotics/Sound-as-Pretext/blob/main/code/data/out/video.gif)](https://www.youtube.com/watch?v=fuexj03mGNo)

### Code

The entire codebase, training scripts and pre-trained models are avaliable [here](https://github.com/idsia-robotics/Sound-as-Pretext/tree/main/code).

### Dataset

The dataset divided into [unlabeled training-set](https://drive.switch.ch/index.php/s/RSz7jRiHrSwf54p), [labeled training-set](https://drive.switch.ch/index.php/s/BfQwbzCf4gTGJ7T), [validation-set](https://drive.switch.ch/index.php/s/qN4NO9296K6ry1t), and [test-set](https://drive.switch.ch/index.php/s/7myEJA7E4zYQlVz) is avaiable through the relative links.

-->
