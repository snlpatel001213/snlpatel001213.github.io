---
layout: page
title: AirControl
description: Using Unity to build something unique for reinforcement learning tasks.
image: assets/images/pic01.jpg
nav-menu: false
description: null
image: null
author: null
show_tile: false
---
<section id="one">
	<div class="inner">
		<header class="major">
			<h1>AirControl</h1>
		</header>
        <h2 id="content">TL;DR</h2>
        <div class="row 200%">
            <div class="12u 12u$(medium)">
            <p>
                Starting this project from scratch on my own and maintaining it provided me with the all togather a next level of experiecne. Here is all I have learnt from this project so far:</p>
                <li>A good command over C# language, refreshed my OOPS concepts</li>
                <li>Designing very complex simulation in unity gmae engine</li>
                <li>Communicating between C# and python by using TCP socket communication</li>
                <li>Developing python API and documenting it</li>
                <li>Documentation of C# by doxygen</li>
                <li>Publishing build for various os including, OSX, Windows and Linux</li>
                <li>Publishing package to pypi</li>

            </div>
        </div>
        <h2 id="content">The Concept</h2>
        <div class="row 200%">
            <div class="8u 12u$(medium)">
                <p>There exist many airplane simulators like - Microsoft flight simulator, GeoFS, etc.. but none provide programming interface. So it one want to do Reinforcement learning task on such platform its not possible. These simulators close source so it makes even difficult to work with. 
                Hence I created an unity based Open Source, Modular, Cross-Platform, and Extensible Flight Simulator For Deep Learning Research that can be control through python. Below are the some of the feature of Aircontrol</p>
                <li>Built with <b>C#</b>, it has <b>Python</b> API to control it from your favourite Deep learning Framework.</li>
                <li>Complete source code is open on Github.</li>
                <li>Aircontrol takes full advantage of object-oriented programming. It is developed fully modular from day one. You can easily introduce new features such as **vertical takeoff**. You can bring your own **alien plane to AirCotrol**. </li>
                <li>AirControl is truly cross-platform and can be compiled on Linux, macOS, and Windows. Binary will be released for all the platforms.</li>
                <li>AirControl uses Nvidia <a href="https://en.wikipedia.org/wiki/PhysX">Phyx</a> for the best possible Newtonian physics simulation.</li>
                <li>AirControl allows users to take advantage of aerodynamic effects such as <a href="https://en.wikipedia.org/wiki/Ground_effect_(aerodynamics)">Ground effect</a>.</li>
                <li>All the control surfaces (Throttle, Rudder, Ailerons, and Flaps) accept normalized input between -1 and 1. This makes AirControl even more friendly with AI.</li>
                <li>Airplane behavior can be finetuned using a configuration file.</li>
                <li>Available for Linux, Windows, Mac, and Linux headless mode.</li>
                <li>Support simultaneous multiple instances at different ports.</li>
            </div>
            <div class="3u 12u$(medium)">
                    <div class="12u"><span class="image fit"  style="object-fit: contain" ><img src="{% link assets/images/aircontrol/logo.png %}" alt="" /></span></div>
                    <div class="12u"><span class="image fit"  style="object-fit: contain" ><img src="{% link assets/images/aircontrol/camera1.png %}" alt="" /></span></div>
                    <div class="12u$"><span class="image fit" style="object-fit: contain" ><img src="{% link assets/images/aircontrol/unity.png %}" alt="" /></span></div>
            </div>
        </div>
        <!-- row -->
        <h2 id="content">Available Airplanes</h2>
        <div class="row 200%">
            <div class="6u 12u$(medium)">
                <p>At present AirControl has two airplanes, F4UCorsair and Cessna152. F4UCorsair about 3X in size and weight compared to Cessna152.
                    Both the airplanes can be configured exteranlly for different flying experience. Both the airplanes behave symetrically when it comes to API level control.
                </p> 
            </div>
            <div class="6u 12u$(medium)">
                <span  class="image fit"  style="object-fit: contain"  ></span><img src="{% link assets/images/Airplanes.png %}" />
            </div>
       </div>
       <!-- row -->
       <h2 id="content">Controlling with python</h2>

        <div class="row 200%">
            <div class="12u 12u$(medium)">
                <p>AirControl can be fully controlled through python. Along with flight other functions like lidar range, capture camera, scene camera, level reset, host port setting, etc.. can be controlled through python.                </p>            
                <b> Following are few of the python client examples:</b>
            </div>
        </div>
        <div class="row 200%">
            <div class="12u 12u$(medium)">
                <table>
                    <thead>
                    <tr>
                    <th>Sr. No.</th>
                    <th>Client Example</th>
                    <th>Details</th>
                    </tr>
                    </thead>
                    <tbody>
                    <tr>
                    <td>1</td>
                    <td><a href="https://github.com/snlpatel001213/AirControl/blob/master/Python/client_examples/input_output_API.ipynb">Input Output Python API</a></td>
                    <td>Commanding AirControl through python</td>
                    </tr>
                    <tr>
                    <td>2</td>
                    <td><a href="https://github.com/snlpatel001213/AirControl/blob/master/Python/client_examples/flight_loop.ipynb">Reinforcment Learning Flight loop</a></td>
                    <td>Example reinforcement learning flight loop</td>
                    </tr>
                    <tr>
                    <td>3</td>
                    <td><a href="https://github.com/snlpatel001213/AirControl/blob/master/Python/client_examples/flight_loop.ipynb">Finetuning Airplane</a></td>
                    <td>Finetune flight experience</td>
                    </tr>
                    <tr>
                    <td>4</td>
                    <td><a href="https://github.com/snlpatel001213/AirControl/blob/master/Python/client_examples/launch_from_python.ipynb">Launch from command line</a></td>
                    <td>Launching Aircontrol from command line</td>
                    </tr>
                    <tr>
                    <td>5</td>
                    <td><a href="https://github.com/snlpatel001213/AirControl/blob/master/Python/client_examples/launch_from_python.ipynb">Launch from python</a></td>
                    <td>Launching Aircontrol from Python</td>
                    </tr>
                    <tr>
                    <td>6</td>
                    <td><a href="https://github.com/snlpatel001213/AirControl/blob/master/Python/client_examples/primitive_API.ipynb">Primitive API</a></td>
                    <td>Simple Client to interact with the server. It does not require the AirControl Pypi package. Just for unit tests, not for long runs</td>
                    </tr>
                    <tr>
                    <td>7</td>
                    <td><a href="https://github.com/snlpatel001213/AirControl/blob/master/Python/client_examples/primitive_API_2.ipynb">Primitive API - 2</a></td>
                    <td>Simple Client to interact with server. More detailed than the previous one. An end-to-end flight loop is demonstrated. It does not require the AirControl Pypi package. Just for unit tests, not for long runs</td>
                    </tr>
                    <tr>
                    <td>8</td>
                    <td><a href="https://github.com/snlpatel001213/AirControl/blob/master/Python/client_examples/lidar_API.ipynb">Lidar Controls</a></td>
                    <td>Demonstrate how to control lidar from the python client.</td>
                    </tr>
                    <tr>
                    <td>9</td>
                    <td><a href="https://github.com/snlpatel001213/AirControl/blob/master/Python/client_examples/camera_API.ipynb">Camera Controls</a></td>
                    <td>Demonstrate how to control Camera from the python client. It allows switching the camera. It allows for capturing Depth, Semantic segmentation, Object segmentation, and Optical flow variant of the scene.</td>
                    </tr>
                    <tr>
                    <td>10</td>
                    <td><a href="https://github.com/snlpatel001213/AirControl/blob/master/Python/client_examples/time_of_day_API.ipynb">Time of Day Controls</a></td>
                    <td>Allows controlling the time of day and light conditions. It allows controlling sun position based on Longitude, Latitude, Hour, and Minutes.</td>
                    </tr>
                    <tr>
                    <td>11</td>
                    <td><a href="https://github.com/snlpatel001213/AirControl/blob/master/Python/client_examples/other_API.ipynb">UI and Audio Controls</a></td>
                    <td>Allows controlling the visibility of Airplane control on UI and Airplane Audio.</td>
                    </tr>
                    </tbody>
                    </table>
            </div>
       </div>
       <!-- row -->
       <h2 id="content">Image Capture</h2>
       <div class="row 200%">  
            <div class="6u 12u$(medium)">
                <div class="box alt">
                    <div class="row 50% uniform">
                        <div class="4u"><span class="image fit"><img src="{% link assets/images/aircontrol/1.png %}" alt="" /></span></div>
                        <div class="4u"><span class="image fit"><img src="{% link assets/images/aircontrol/2.png %}" alt="" /></span></div>
                        <div class="4u$"><span class="image fit"><img src="{% link assets/images/aircontrol/3.png %}" alt="" /></span></div>
                        <!-- Break -->
                        <div class="4u"><span class="image fit"><img src="{% link assets/images/aircontrol/4.png %}" alt="" /></span></div>
                        <div class="4u"><span class="image fit"><img src="{% link assets/images/aircontrol/5.png %}" alt="" /></span></div>
                        <div class="4u$"><span class="image fit"><img src="{% link assets/images/aircontrol/6.png %}" alt="" /></span></div>
                        <!-- Break -->
                        <div class="4u"><span class="image fit"><img src="{% link assets/images/aircontrol/7.png  %}" alt="" /></span></div>
                        <div class="4u"><span class="image fit"><img src="{% link assets/images/aircontrol/8.png %}" alt="" /></span></div>
                        <div class="4u$"><span class="image fit"><img src="{% link assets/images/aircontrol/9.png %}" alt="" /></span></div>
                    </div>
                </div>    
            </div>
            <div class="6u 12u$(medium)">
                <p>One of the main challenges in Machine Learning is the task of getting large amounts of training data in the right format. Deep learning, and machine learning more generally, needs huge training sets to work properly. Virtual worlds can provide a wealth of training data. However, it must consist of more than just the final image: object categorization, optical flow, etc
                </p>
                <p>Aircontrol allow capturing following image types from all camera attached to the airplanes</p> 
                <div class="table-wrapper">
                    <table>
                            <tr>
                                <td>Capture Type</td>
                                <td>Type</td>
                                <td>Details</td>
                            </tr>
                            <tr>
                                <td>0</td>
                                <td>Scene Capture</td>
                                <td>Capture from scene Camera</td>
                            </tr>
                            <tr>
                                <td>1</td>
                                <td>Instance Segmentation</td>
                                <td>Each object in the scene gets unique color</td>
                            </tr>
                            <tr>
                                <td>2</td>
                                <td>Semantic segmentation</td>
                                <td>Objects are assigned color based on their category</td>
                            </tr>
                            <tr>
                                <td>3</td>
                                <td>Depth</td>
                                <td>Pixels are colored according to their motion in the relation to the camera</td>
                            </tr>
                            <tr>
                                <td>4</td>
                                <td>Normals</td>
                                <td>Surfaces are colored according to their orientation in relation to the camera</td>
                            </tr>
                            <tr>
                                <td>5</td>
                                <td>Optical Flow</td>
                                <td>Pixels are colored according to their distance from the camera (Only visible when the Airplane or Object in reference is moving)</td>
                            </tr>
                    </table>
                </div>
            </div>
       </div>
       <!-- row -->
       <h2 id="content">Lidar</h2>
        <div class="row 200%">
            <div class="6u 12u$(medium)">
                <p>The Aircontrol Lidar API facilitates communication with simulated LIDAR in Unity.
                    The lidar script uses the Raycast feature of Unity Game Engine. The distance array, of float type, consists of 360 members, the ith member storing the distance at which the ray hit an object at the ith degree.</p> 
            </div>
            <div class="6u 12u$(medium)">
                <span  class="image fit"  style="object-fit: contain"  ></span><img src="{% link assets/images/aircontrol/lidar.png %}" />
            </div>
       </div>
        <div class="row 200%">
            <div class="12u 12u$(medium)">
                <p>There are tons of such features present in the AirControl. Please visit the github page.</p>
            </div>
        </div>
        <h2 id="content">Releases</h2>
        <div class="row 200%">
            <div class="12u 12u$(medium)">
                <p>Aircontrol builds are regularly released for Windows, Linux, Mac, wegGL and Linux Headless</p>
                <p>Please visit to release page to play with the build :  <a href="https://github.com/snlpatel001213/AirControl/releases" target="blank_">AirControl Releases</a> </p>
            </div>
        </div>
        <h2 id="content">Github</h2>
        <div class="row 200%">
            <div class="12u 12u$(medium)">
                <span  class="image fit"  style="object-fit: contain" ></span><a href="https://github.com/snlpatel001213/AirControl" target="blank_"><img src="{% link assets/images/github_front.png %}" /></a>
            </div>
        </div>
        <h2 id="content">Python API Documentation</h2>
        <div class="row 200%">
            <div class="6u 12u$(medium)">
                <span  class="image fit"  style="object-fit: contain"  ></span><a href="https://aircontrol.readthedocs.io/" target="blank_"><img src="{% link assets/images/rtd_front.png %}" /></a>
            </div>
            <div class="6u 12u$(medium)">
                <p>Detailed Python API documentation built using sphinx</p>
            </div>
        </div>
        <h2 id="content">C# API Documentation</h2>
        <div class="row 200%">
            <div class="8u 12u$(medium)">
                <span  class="image fit"  style="object-fit: contain"  ></span><a href="https://snlpatel001213.github.io/AirControl/html/index.html" target="blank_"><img src="{% link assets/images/csharp_front.png %}" /></a>
            </div>
            <div class="4u 12u$(medium)">
                <p>Detailed C# API documentation built using doxygen</p> 
            </div>
        </div>
   </div>

        
</section>
  
