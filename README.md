# Self-Supervised Learning of Visual Robot Localization Using LED State Prediction as a Pretext Task

*Mirko Nava, Nicholas Carlotti, Luca Crupi, Daniele Palossi, and Alessandro Giusti*

Dalle Molle Institute for Artificial Intelligence, USI-SUPSI, Lugano (Switzerland)

### Abstract

We propose a novel self-supervised approach for learning to localize robots equipped with controllable LEDs visually. 
We rely on a few training samples labeled with position ground truth and many training samples in which only the LED state is known, whose collection is cheap. We show that using LED state prediction as a pretext task significantly helps to solve the visual localization end task.
The resulting model does not require knowledge of LED states during inference. <br>
We instantiate the approach to visual relative localization of nano-quadrotors: experimental results show that using our pretext task significantly improves localization accuracy (from 68.3% to 76.2%) and outperforms alternative strategies, such as a supervised baseline, model pre-training, or an autoencoding pretext task. We deploy our model aboard a 27-g Crazyflie nano-drone, running at 21 fps, in a position-tracking task of a peer nano-drone.
Our approach, relying on position labels for only 300 images, yields a mean tracking error of 4.2 cm versus 11.9 cm of a supervised baseline model trained without our pretext task.

<img src="https://github.com/idsia-robotics/leds-as-pretext/blob/main/img/led_pretext_approach.png" width="850" alt="LEDs as Pretext approach" />

Figure 1: *Overview of our approach. A fully convolutional network model is trained to predict the drone position in the current frame by minimizing a loss **L**end defined on a small labeled dataset **T**l (bottom), and the state of the four drone LEDs, by minimizing **L**pretext defined on a large dataset **T**l joined with **T**u (top).*

<br>

Table 1: *Comparison of models, five replicas per row. Pearson Correlation Coefficient ρu and ρv , precision P30 and median of the error D tilde.*

<img src="https://github.com/idsia-robotics/leds-as-pretext/blob/main/img/led_pretext_performance.png" width="900" alt="LEDs as Pretext performance" />

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

[![Self-Supervised Learning of Visual Robot Localization Using Prediction of LEDs States as a Pretext Task](https://github.com/idsia-robotics/leds-as-pretext/blob/main/img/led_pretext_video_preview.gif)](https://github.com/idsia-robotics/leds-as-pretext/blob/main/img/led_pretext_video.mp4?raw=true)

### Code

The codebase is avaliable [here](https://github.com/idsia-robotics/leds-as-pretext/tree/main/code).

-->

### Supplementary Material

```diff
! Videos and code of the proposed approach will soon be made avialable to the community !
```
