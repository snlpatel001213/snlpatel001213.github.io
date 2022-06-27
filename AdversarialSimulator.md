---

layout: page
title: Adversarial Simulator
description: Adversarial Simulator allows one to simulate adversarial attacks in the virtual environment within the Unreal Engine and inference with Nvidia DRIVE SDK.
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
			<h1>Adversarial Simulator</h1>
		</header>
        <h2 id="content">TL;DR</h2>
        <div class="row 200%">
            <div class="12u 12u$(medium)">
            <p>
                This is the very first project when I started learning about 3D game engines and how to use them for deep learning. Here is what I learned : 
           
                <li>Developing Games with C++</li>
                <li>Generating realistic data from Unreal engine</li>
                <li>Setting up reproducible experimentation in unreal engine</li>
                <li>Blackbox and Whitebox adversarial attack</li>
            </p>
            </div>
        </div>
        <h2 id="content">The Concept</h2>
        <div class="row 200%">
            <div class="12u 12u$(medium)">
                <p>-This project was developed to strudy the effect of adversarial attack environment must be controlled. Autonomous driving includes mainly 5 phases sense, perceive, map, plan, and Drive. Autonomous Vehicle ‘sense’ the surrounding with the help of Cameras and Lidars. Deep Learning techniques are considered Blackbox and found to be vulnerable to adversarial attacks. In this research, we study the effect of the various known adversarial attack in the Unreal Engine-based high-fidelity real-time raytraced simulated environment. This experiment seeks an answer to questions such as if adversarial attacks work in moving vehicle scenarios and can an unknown network be targeted. We found that the existing Blackbox and Whitebox affects different traffic signs differently. Attack found to be affecting classification in static scenes are not similarly affecting in the moving vehicle scenarios. However, some attacks with barely noticeable perturbations were found to completely block the identification of certain traffic signs. By simulating the interaction of light on Traffic sign have also observed that daylight condition has a significant impact on the performance of the model. Our findings are found to be closely simulating scenarios as found in the real world.</p>
            </div>
        </div>
        <!-- row -->
        <h2 id="content">Available Environments</h2>
        <div class="row 200%">
            <div class="12u 12u$(medium)">
                <p>Adversarial simulator supports Environment so that different scenarios can be generated. In Unreal Engine, such scenarios are organized in Maps. adversarial Simulator provides two maps.</p> 
            </div>
        </div>
        <div class="row 200%">
            <div class="6u 12u$(medium)">
                <p>A map of the test facility developed within Unreal Engine. The diagram shows a test track for one traffic sign. All the traffic signs are independently arranged in parallel in a non-occluding fashion. High fidelity environment with Real-time ray tracing simulating shadows and reflection. Fine details like objects 1) cars and buildings 2) Traffic sign placed at the end of the track 3) Holdings 4) detailed rocks and vegetation with shadows 5) Realistic vegetation 6) Asphalt like material for the road with lane markings and 7) water reflecting sun rays.</p>    
            </div>
            <div class="6u 12u$(medium)">
                <span  class="image fit"  style="object-fit: contain"  ></span><img src="{% link assets/images/adversarialsimulator/map1.png %}" />
            </div>
        </div>
        <div class="row 200%">
            <div class="6u 12u$(medium)">
                <p>The second environment is comparatively bigger than the previous one. This environment has  various kinds of assets including
                    <ul>
                        <li>Two lanes and one lane roads</li>
                        <li>Uneven terrain</li>
                        <li>water bodies</li>
                        <li>sky-scrappers</li>
                        <li>Small houses</li>
                        <li>Retail stores</li>
                        <li>Variety of vehicles</li>
                        </ul>
                </p>    
            </div>
            <div class="6u 12u$(medium)">
                <div class="box alt">
                    <div class="row 50% uniform">
                        <div class="4u"><span class="image fit"><img src="{% link assets/images/adversarialsimulator/map2_1.png %}" alt="" /></span></div>
                        <div class="4u"><span class="image fit"><img src="{% link assets/images/adversarialsimulator/map2_2.png %}" alt="" /></span></div>
                        <!-- Break -->
                        <div class="4u"><span class="image fit"><img src="{% link assets/images/adversarialsimulator/map2_3.png %}" alt="" /></span></div>
                        <div class="4u"><span class="image fit"><img src="{% link assets/images/adversarialsimulator/map2_4.png %}" alt="" /></span></div>
                    </div>
                </div>    
            </div>
        </div>
       <!-- row -->
       <h2 id="content">Assets</h2>

        <div class="row 200%">
            <div class="6u 12u$(medium)">
                <div class="box alt">
                    <div class="row 50% uniform">
                        <div class="4u"><span class="image fit"><img src="{% link assets/images/adversarialsimulator/map3_1.png %}" alt="" /></span></div>
                        <div class="4u"><span class="image fit"><img src="{% link assets/images/adversarialsimulator/map3_2.png %}" alt="" /></span></div>
                        <!-- Break -->
                        <div class="4u"><span class="image fit"><img src="{% link assets/images/adversarialsimulator/map3_3.png %}" alt="" /></span></div>
                        <div class="4u"><span class="image fit"><img src="{% link assets/images/adversarialsimulator/map3_4.png %}" alt="" /></span></div>
                    </div>
                </div>    
            </div>
            <div class="6u 12u$(medium)">
                <p>There are various add ons provide various assets out of these few are suggested here:</p>
                <ul>
                    <li>Modular road: To build the road and bridges</li>
                    <li>Animal Variety pack: Animals, can be commonly used for obstacle detection models</li>
                    <li>Scanned 3D People Pack: Fully driveable - Vehicles like cars and trucks</li>
                    <li>City Object Pack: It has various objects like signboards, dustbins, flag mannequin etc.</li>
                </ul>
            </div>
        </div>
       <!-- row -->
       <h2 id="content">Github</h2>
       <div class="row 200%">  
        <p>Research paper for this project is is accepted at Waset ICCV 2022. Following is the link to github page for more info</p>
            <div class="12u 12u$(medium)">
                <div class="12u">
                    <span class="image fit">
                        <a href="https://github.com/snlpatel001213/AdversarialSimulator" target="blank_">
                                    <img src="{% link assets/images/adversarialsimulator/github.png %}" alt="" />
                        </a>
                    </span>
                </div>
            </div>
       </div>
       
   </div>

        
</section>
  
