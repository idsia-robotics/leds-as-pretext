# Self-Supervised Learning of Visual Robot Localization Using Prediction of LEDs States as a Pretext Task

*Mirko Nava, Nicola Armas, Antonio Paolillo, Jerome Guzzi, Luca Maria Gambardella, and Alessandro Giusti*

Dalle Molle Institute for Artificial Intelligence, USI-SUPSI, Lugano (Switzerland)

### Abstract

We propose a novel self-supervised approach to learn CNNs that perform visual localization of a robot in an image using very small labeled training datasets. Self-supervision is obtained by jointly learning a pretext task, i.e., predicting the state of the LEDs of the target robot.
This pretext task is compelling because: a) it indirectly forces the model to learn to locate the target robot in the image in order to determine its LED states; b) it can be trained on large datasets collected in any environment with no external supervision or tracking infrastructure.
We instantiate the general approach to a concrete task: visual relative localization of nano-quadrotors.
Experimental results on a challenging dataset show that the approach is very effective; compared to a baseline that does not use the proposed pretext task, it reduces the mean absolute localization error by as much as 78% (43 to 9 pixels on *x*; 28 to 6 pixels on *y*).


![LEDs as Pretext](https://github.com/idsia-robotics/leds-as-pretext/blob/main/img/led_pretext_approach.png)
Figure 1: *Overview of our approach. The model is trained to predict: the drone position in the current frame, by minimizing the end loss (**L**end) defined on **T**l (bottom); and the current state of the four drone LEDs, by minimizing the pretext loss (**L**pretext) defined on **T**l and **T**u (top).*

![LEDs as Pretext](https://github.com/idsia-robotics/leds-as-pretext/blob/main/img/led_pretext_performance.png)
Figure 2: *On the left, comparison of approaches in terms of MAE (lower is better) and R2 score (higher is better) for the x and y variables.
On the right, comparison of baseline (red), LEDs as a Pretext (green), and Upper Bound (blue) models trained with varying amounts of labels. MAE improvement refers to the percentage reduction in MAE between baseline and our LED-P approach. Results obtained by averaging the performance on the x and y variables.*

<!--
The PDF of the article is available in Open Access [here]( https://doi.org/10.1109/LRA.2022.3143565).

### Bibtex

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
