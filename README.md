---
layout: default
title: {{ site.name }}
---

<div id="home">
  <div id="site">
    <h2>Check out the enhanced version of this presentation at <a href="https://radudilirici.github.io/autonomous-pong-robot/">https://radudilirici.github.io/autonomous-pong-robot/</a>.</h2>
  </div>

  <div id="summary">
    <h2>Summary</h2>

    <p>This is my Bachelor thesis. It consists of a small robot that can play the game Pong by physically controlling
      one of the available joysticks.</p>
    <p>The game is run and displayed on a <i>Raspberry Pi</i>. There are two joysticks attached to it that the players
      can use in order to control the paddles.</p>
    <p>
      The robot observes the state of the game through a camera and uses <b>Computer Vision</b> to extract the relevant
      data.
      Based on the processed information, it then computes what actions to play in order to win.
      To act out the desired strategy, the robot operates one of the joysticks through a custom arm.
    </p>
    <p>
      The project includes three separate methods for determining the robot decisions.
      The first one uses conventional programming and the other two are built using <b>Reinforcement Learning</b>
      algorithms.
    </p>
    <p>
      The source files can be found at my <a href="{{ site.github_repo }}">GitHub
        project</a>.
    </p>
  </div>

  <div id="demo">
    <h2>Demo</h2>

    <figure>
      <video width="852" height="480" autoplay muted loop>
        <source src="resources/demo.mp4" type="video/mp4">
        Your browser cannot play this video.
      </video>
      <figcaption>The robot, controlling the right joystick, playing against a person.</figcaption>
    </figure>
  </div>

  <div id="pong">
    <h2>Pong</h2>

    <p>
      For this project, a custom version of Pong that runs on <i>Raspberry Pi</i> was implemented.
      The game is shown on a screen and it can be played using two joysticks.
    </p>
    <p>
      The game is written in <b>Python 3</b> and uses Pygame.
      It also takes advantage of <b>multithreading</b> in order to run at a consistent 60 frames per second.
    </p>
    <p>
      The Pong game runs on a custom hardware setup that was built specifically for this project.
      The circuit design can be observed in the next figure.
    </p>

    <figure>
      <img src="resources/hardware.png" alt="The hardware circuit.">
      <figcaption>The hardware system of the Pong game.</figcaption>
    </figure>
  </div>

  <div id="processing">
    <h2>Image Processing</h2>

    <p>
      The robot observes the state of the game through a small camera. This continuous flow of images has to be
      processed before applying any game strategy.
      This is a common step in all three algorithms. It was written in Python 3 and uses <a
        href="https://docs.opencv.org/4.5.4/index.html">OpenCV</a>.
    </p>
    <figure>
      <img src="resources/processing.png" alt="Input processing flow.">
      <figcaption>
        The process of cropping the image to the contents of the game.
        <br />
        The highlighted points represent the detected corners.
      </figcaption>
    </figure>
    <p>
      First, a binary threshold is applied on the input frame.
      Based on the new image, the corners of the game are extracted using the contours of the objects.
      Then, a four point perspective transformation is applied to the area inside the corners.
    </p>
    <p>
      By applying this process, the robot can process images even if it views the game from a bad angle.
      For performance reasons, the corners are not recomputed on every iteration.
      This detection is reapplied from time to time in case the camera or the display is moved.
    </p>
  </div>

  <div id="algorithms">
    <h2>Algorithms</h2>

    <p>
      The project includes three algorithms for the robot strategy. The first one consists of conventional
      programming, and the other two are based on <b>Deep Q Learning</b> which is a form of Reinforcement
      Learning.
    </p>

    <div id="conventional">
      <h3>The conventional method</h3>

      <p>
        The first algorithm uses classic programming in order to find out the optimal strategy.
        After the initial image processing, the contours of the objects are extracted and their centers are computed.
      </p>
      <p>
        The program uses the history of the ball positions in order to calculate it's current trajectory.
      </p>

      <figure>
        <video width="850" height="228" autoplay muted loop>
          <source src="resources/detection_05_cropped.mp4" type="video/mp4">
          Your browser cannot play this video.
        </video>
        <figcaption>
          Data extraction example in slow motion (x0.5).
          <br />
          The paddles are marked with green dots, while the ball is labeled with red.
          <br />
          The yellow dot represents the prediction of where the ball will end up.
        </figcaption>
      </figure>

      <p>
        With this new information available, the program can now decide it's next move.
        If the ball is moving away from the robot paddle, it tries to get to the middle of the screen.
        Otherwise, the robot should aim to move to the predicted position.
      </p>

      <p>
        Because of the limitations of the hardware components, namely the servomotor, the robot actions suffer some
        delay.
        To compensate for this, the program uses some tricks in order to take into account the resulting "inertia".
      </p>

      <p>
        This algorithm yields the best results out of all three.
      </p>
    </div>

    <div id="dqn-1">
      <h3>Deep Q Learning</h3>

      <p>
        In order to train the models, simulated replicas of the game were built.
        This was essential, so the training was not limited by the speed of the real Pong game.
      </p>

      <p>
        One of the biggest difficulties encountered with the Reinforcement Learning approach was the difference between
        the simulated environment and the real life scenario.
        One such example is the delay of the robotic arm. To ameliorate it's impact, this delay was included in the
        simulation, so that the models can learn to take it into consideration.
      </p>
    </div>

    <div id="dqn-1">
      <h3>Deep Q Learning on images</h3>

      <p>
        The first algorithm that uses Reinforcement Learning operates directly on the images of the game.
        For this, a custom <b>Convolutional Neural Network</b> was trained on the Pong replica using <b>PyTorch</b>.
      </p>

      <figure>
        <img src="resources/nn_1.png" alt="CNN architecture">
        <figcaption>
          The architecture of the Convolutional Neural Network.
          <br />
          The output of the network are the three possible actions: move up, down, or stay.
        </figcaption>
      </figure>

      <p>
        This method doesn't have such good results, but it's the most versatile, as the robot learns everything on it's
        own.
        This way, no previous knowledge of the game is required by the programmer.
      </p>
    </div>

    <div id="dqn-2">
      <h3>Deep Q Learning on object positions</h3>

      <p>
        This method combines the two previously discussed approaches.
        The coordinates of the objects are extracted and fed to the Reinforcement Learning model.
        This way, it can create it's own strategy using
      </p>

      <p>
        This provided several advantages over the first RL algorithm.
        The input was easier for the model to interpret and the size and complexity of the neural network decreased
        significantly.
      </p>

      <figure>
        <img src="resources/nn_2.png" alt="NN architecture" width="600">
        <figcaption>
          The architecture of the new network.
        </figcaption>
      </figure>

      <p>
        This program results in a better player than the previous Q Learning method, but not as good as the
        deterministic algorithm.
      </p>
    </div>
  </div>

  <div id="conclusion">
    <h2>Conclusion</h2>

    <p>
      While the classic algorithm outperforms both Reinforcement Learning agents, it is based on the knowledge of the
      programmer.
    </p>
    <p>
      Moreover, it's functionality is entirely dependent on the specific version of Pong.
      Although the robot might be able to play some other game implementations, it would most likely require adjustments
      to properly process the input.
    </p>
    <p>
      The intelligent methods would also have to learn about other Pong versions, but the process would be more
      automatic.
      Futhermore, the resulting program would likely be the most capable one of playing a new version of the game.
    </p>
  </div>
</div>