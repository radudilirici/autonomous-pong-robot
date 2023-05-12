<div id="home">
  <h2>Summary</h2>
  <div>
    <p>This is my Bachelor thesis. It consists of a small robot that can play the game Pong by physically controlling one of the available joystics.</p>
    <p>The game is run and displayed on a <i>Raspberry Pi</i>. There are two joysticks attached to it that the players can use in order to controll the paddles.</p>
    <p>
      The robot observes the state of the game through a camera and uses <b>Computer Vision</b> to extract the relevant data.
      Based on the processed information, it then computes what actions to play in order to win.
      To act out the desired strategy, the robot operates one of the joysticks through a custom arm.
    </p>
    <p>
      The project includes three separate algorithms for determining the robot decisions.
      The first uses conventional programming and the other two take advantage of custom <b>Machine Learning</b> models.
    </p>
  </div>

  <h2>Demo</h2>
  <video width="852" height="480" autoplay muted loop>
    <source src="resources/demo.mp4" type="video/mp4">
    Your browser does not support this video.
  </video>
</div>
