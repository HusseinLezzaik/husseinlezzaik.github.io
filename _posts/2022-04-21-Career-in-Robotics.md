---
layout: post
comments: true
title: "Career in Robotics: Perspective and Reflections"
excerpt: "After struggling with ROS, my whole philosophy on robotics engineering became software engineering driven."
date:   2022-04-21 08:00:00
mathjax: false
---

Almost six years ago, I started my career in robotics and to establish myself in the field, I worked really hard on building side projects & participating in hackathons, as well on both research and industrial level robotics projects. A lot has changed in my perspective since then, with lots of skills and tools learned, and assumptions unlearned.

I keep getting asked questions like “How does it look like to work on robots in the real world?”, noticing that there’s actually a big gap between what robotics looks like in schools and hackathons, versus building actual products for customers in the real world.

It took me quite a while to get on track in my first role in the industry, and it was very stressful to catch up with everything even though I thought that I was already prepared to build robots in the real world like most students think when trained at school.

Over the past year, my perspective has matured a lot on what it actually looks like to build a career in robotics, and I am writing this article to share what I struggled to learn the hard way for people that are still getting into this field or at school dreaming to work in robotics one day!

`Disclaimer #1:` This article focuses on the engineering side of robotics, and doesn't get too much into robotics research.

`Disclaimer #2:` The field is evolving very fast, and by the time you see robots in the streets as part of your daily life, probably a lot of the content of this article will become out-dated!

## Table of contents:
- [What is a Robotics Engineer?](#what-is-a-robotics-engineer)
- [Software Engineering](#software-engineering)
    - [Python](#python)
    - [C++](#c)
    - [ROS](#python)
    - [Version Control](#version-control)
    - [Object Oriented Programming](#object-oriented-programming)
    - [CI/CD & Testing](#cicd--testing)
    - [Software Architectures](#software-architecture)
    - [Containerization](#containerization)
    - [Networking](#networking)
    - [Cloud Services](#cloud-services)
    - [Shell and Linux](#shell-and-linux)
    - [Computer Architectures](#computer-architectures)
    - [Applied Machine Learning](#applied-machine-learning)
    - [Simulators](#simulators)
- [Roboticist](#roboticist)
    - [Algorithms](#algorithms)
        - [Perception](#1-perception)
        - [Planning](#2-planning)
        - [Control](#3-control)
    - [Control Theory](#control-theory)
    - [Machine Learning](#machine-learning)
    - [Modeling](#modeling)
    - [Systems Engineering](#systems-engineering)
- [Applied Scientist](#applied-scientist)
    - [Research and Development Skills](#research--development-skills)
    - [Research Experience](#research-experience)
    - [Publications](#publications)
- [Conclusion](#conclusion)


## What is a Robotics Engineer? 
A robotics engineer is essentially a software engineer (SWE), with domain knowledge in robotics algorithms & hardware, and an applied scientist by heart.

Robotics Engineer = Software_Engineer + Robotics_Background + Applied_Scientist

It might seem like an irony at first, but most of what you’ll actually be doing in the industry is core software engineering, rather than algorithmic development for robotics from scratch. Robots, as Elon Musk puts it, are “computers on wheels”. Therefore, to build such systems end-to-end (E2E) that are actually functional, reliable, and accurate to the degree of driving humans on roads requires a lot of hard engineering.

Writing software for robots means building huge code bases that should be structured in a clean, scalable and reusable way. Much like any SaaS industry.

Unlike hackathons or research projects, usually code quality isn’t a main factor into the equation, and you can make as many assumptions as you want to prove that your algorithm is SOTA on a benchmark or tweak parts of your code to give a quick MVP demo to a jury.

However, in the real world you’ll be scratching your head on things like making your algorithms faster and lighter on memory/compute for inference, or writing code to monitor your battery level and making sure your robot doesn’t get stuck in a tight corner.

Therefore a big portion of your job is going to be pure software engineering, and it’s going to be expected that you can operate like one!

## Software Engineering
Writing software for robotics is inherited from the best practices of building software from other disciplines, and is still witnessing ongoing progress. Therefore, it’s best to first learn the fundamentals of building production level software before jumping into the specifics of robotics software.


Lots of acquired skills from software engineering roles are easily applicable and transferable into robotics projects. [Andrej Karpathy](https://karpathy.ai/), the head of AI at Tesla for example, came from a computer science background during his undergrad and later on got more into autonomous systems.

If you’re a strong SWE, learning the robotics stuff is easier than knowing the robotics stuff and then training to become an engineer. So if you’re still early in your career, it’s better to focus on sharpening your SWE skills and grasping the concepts of the tools you are working with.

Here’s a non-exclusive list of the most fundamental tools and concepts that are important for building robotics software:

### Python
Probably doesn’t need introduction these days, but it’s widely used across the robotics community & lots of off-the-shelf algorithms/libraries for robotics are written in Python.

It’s popularity is because of its very fast prototyping feature, and easiness to both learn and write code with. Expect to write a lot of Python code during your job as a robotics engineer!

### C++
Wanna run algorithms on the edge in real time? C++ is your best friend. Lots of robotics frameworks are written in C++, especially those that need to process data online and make predictions in real time.

The reason is that C++ is much faster than Python code during inference, and object oriented friendly. It’s widely used in research labs and also big tech companies in the industry like Amazon Robotics. Investing in mastering C++ will pay off in your career.

### ROS
[ROS](http://wiki.ros.org/) is probably one of the most special software tools that you might not hear of outside robotics. ROS is the Robot Operating System, it’s a microservices architecture built around a publish/subscribe protocol.

ROS is the way you’ll program robots, read data from sensors, publish control commands to your motors, and even use in simulations to test your algorithms that are being developed. 

ROS is quite silly to grasp at the beginning, but once you get a hang of it, it’s going to be one of the easiest parts of your job. Try to master this framework, you can either practice ROS in simulators like [Gazebo](http://gazebosim.org/) or with robot toolkits like [Jetbot](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetbot-ai-robot-kit/).

Now ROS isn’t the only framework out there, but it’s the most popular and if you understand it then you can easily learn other tools when you need to. There’s other alternatives like Lightweight Communication (LCM), and you can even build your [own](https://autorob.org/) robotics operating system if you want too!

### Version Control
Version control is the practice of tracking and managing changes to software code. Since SWE is a core part of developing robotics, managing different branches & versions of software from features to production is essential for robotics projects.

The most popular tool used for version control is Git, and a few big companies like Meta for example built their own custom version control tools. Invest in mastering Git.

### Object Oriented Programming
Most libraries for mobile/aerial robotics are structured in simple, reusable pieces of code using classes. It’s a way to model robots in code with their underlying characteristics and sensors.

### CI/CD & Testing
When writing code for robots, you want to scale your code base in an incremental and continuous way, so that your system doesn’t have glitches when you’re at advanced stages of development.

Therefore at each stage there’s different types of tests  like unit or integration tests that you need to write or do to make sure your code and system is fully functional.

Want to use the same code for 100s of identical robots? You’ll probably need to write automation tests to make sure your product is ready to ship for customers!

These tests are inherited from the popular [V-Model](https://en.wikipedia.org/wiki/V-Model) of engineering projects. You always want to combine agile & waterfall practices when managing robotics projects.

### Software Architecture
This part of the robotics pipeline is, in my opinion, probably one of the least developed yet and has lots of room for improvement. I believe when more robotics companies are able to find product-market fit,  we’ll see a wave of Robotics-Ops for tooling and infrastructure of robotics software, similar to what the industry is witnessing right now in the MLOps era.

Topics like design patterns, systems design and scalable infrastructure are critical when scaling projects code base and applications.

### Containerization
Containerization is the art of packaging together software code with all its components like libraries, frameworks, and other dependencies so that they can run in their own environment regardless of the machine vendor or if it’s on the cloud. 

Until now [Docker](https://docs.docker.com/) seems to be dominant in the market of containerization, and for good reasons maintaining a containerized software culture for robotics teams pays well in the long term when products scale & computing envs start to change/grow.

### Networking
This is something I learned the hard way. As part of your robotics job you’ll need to work on remote computers via ssh-ing, distribute computation over several machines, get multiple robots to communicate with each other and collaborate, etc.

You can probably learn most of these on the fly, but it’d be useful if you at least had an idea at the beginning on what to expect.

### Cloud Services
A lot of robotics products these days are part of the IoT industry, with data exploding all over the place. Hence a lot of companies invest into connecting their products into the cloud, for software updates and data collection or even model training (for example [Fleet training](https://youtu.be/FnFksQo-yEY?t=807) for vision at Tesla).

### Shell and Linux
Being a computer wizard is actually a very useful skill that you’ll find important during your job. Mastering shells like Bash and knowing the ins and outs of Linux services to utilize them with your algorithms is handy.

One of my favorite courses out there that covers these practical computer skills that aren’t usually taught at school is this course from MIT [The Missing Semester of Your Computer Science Education](https://missing.csail.mit.edu/).  

### Computer Architectures
You won’t probably need to design a computer yourself as a robotics engineer, but you’ll continuously run into things like computer system requirements and operational performance. 

Knowing a bit about the main fundamentals that have to do with computers and memory is actually useful.

### Applied Machine Learning
Neural nets are the go-to algorithms for lots of robotics tasks like vision. Tesla for example is a big player in the [applied AI](https://www.tesla.com/AI) world for autonomous vehicles.

Writing an algorithm to classify images is one thing, building a complete infrastructure from data collection to labeling and model deployment/monitoring is another. 

Mastering the MLOps life cycle is very important when you want to adapt applied machine learning algorithms into your software pipeline.

One of my favorite resources out there on this topic is the [Full Stack Deep Learning](https://fullstackdeeplearning.com/) course from UC Berkeley.

### Simulators
Since some experiments are very expensive and risky to try out on actual robots, using simulations make it possible to rapidly test algorithms, design robots, and train AI algorithms using synthetic data.


Simulators like [CoppeliaSim](https://www.coppeliarobotics.com/) are robust physics-based engines, with high-quality graphics and convenient graphical interfaces for visualization. Best of all, it’s very easy to port your 3D CAD design inside them to experiment with your exact robot!

## Roboticist
This part of the job is where domain expertise comes in, and having a strong grasp of these concepts from first principles is what enables engineers to actually innovate and invent creative robotic products that work!

Building robots as an analogy is like building human beings, with the three following most core parts:

1) The Brain: Computer Science
2) The Nervous System: Electronics and Electrical Engineering
3) The Body: Mechanical Engineering

### Algorithms
Aside from the SWE side of robotics, the domain expertise is in the algorithms that you write to develop the autonomous systems software stack. 

It’s probably quite famous, but knowing the ins and outs of the perception, planning, and control cycle is very important to build E2E systems. One of my favorite online courses out there that cover these main algorithms is the [Self Driving Cars Specialization](https://www.coursera.org/specializations/self-driving-cars) from the University of Toronto.

<img src="/assets/past/self_driving.png" width="100%" />

You can also take a look [here](https://github.com/protontypes/awesome-robotic-tooling) at this compiled list of software packages and tools available for different parts of the software pipeline for robots.

### 1. Perception
Perception algorithms is the way robots understand and map the environment from sensors into space representations that it can comprehend and reason about its surroundings in order to take actions.

You can take a look at the [Vector Space](https://youtu.be/j0z4FweCy4M?t=2945) that Tesla has developed to have a better idea of what it looks like to actually perceive the world from a robots perspective.

This part of the pipeline is where usually cameras or lidars come in, and methods like object detection and tracking, semantic segmentation, and depth estimation using either traditional image processing approaches or deep learning methods like convolutional neural networks.

### 2. Planning
The motion planning part of the pipeline is how the robot calculates its trajectory both locally and globally in order for it to reach its end goal position.  

This is where topics like finding the shortest path over an occupancy grid using algorithms like Dijkstra or RRT show up. If you want to look at a more practical example, check out this package called [move base](http://wiki.ros.org/move_base) from ROS for building up a navigation stack.

### 3. Control
This final part of the pipeline is how you end up sending control commands to the driver of the robot in order to control it to take it into its end goal. More on control in the section below.

## Control Theory
Want to control robots and make them do tasks accurately? Cool, control theory is your way to go. Although for most small applications, you might get away with only using PID controllers and some hyper-parameter tuning.

However, once you get into highly non-linear dynamic systems or [under-actuated systems](https://underactuated.mit.edu/), you're probably going to need advanced algorithms to be able to control your robots.

### Machine Learning
As one of my senior friends pointed out, not mentioning machine learning will probably piss off a lot of people. Knowing the fundamentals of training neural networks is essential when building algorithms for robots. You can check out this cool [blog](https://lavanya.ai/2019/08/10/training-a-neural-network-start-here/) from Lavanya at W&Bs to get started.

### Modeling
Robots usually have dynamic models (aka differential equations) that describe and predict their behaviors, and are usually derived from fundamental laws from scratch.

Knowing the details of the derivations might not be very important practically, but knowing the main models like the [differential drive equations](https://www.gnotomista.com/ancillaries.html#hemingwayianunicycle) for mobile robots to transform the outputs of control algorithms into inputs of the robot wheels will be important when you deploy robots and want to control them in real time.

### Systems Engineering
Since robots are essentially computers on wheels, it’s important to know what sensors are widely adopted in the industry both from an operational and cost-effective perspective, plus the computer chips available.

You can look here at the [Nvidia](https://www.nvidia.com/en-us/self-driving-cars/drive-platform/hardware/) hardware list for self-driving cars for example, to see the limits of compute out there that you want to run your algorithms on.

Also, general familiarity with computing and GPUs is important. Lots of production chips like the Jetson Nano for example have onboard GPUs on them to render graphics from vision sensors for example. 

So knowing the fundamentals of these and choosing specs according to your system requirement will be something you might need to do oftenly.

Finally, stuff like hardware assembling and mechatronics in general is important.

## Applied Scientist
This one is probably the least heard of, and might sound surprising at the beginning! What does research have to do with robotics in the real world and outside of academia?
Actually it’s very important for robotics, and a lot! A lot of the go-to algorithms that are being used in the industry are coming from academia, like SLAM  algorithms for example. 

And aside from following research literature and algorithms, approaching robotics projects in particular requires a research and development mindset.

### Research & Development Skills
A lot of the work you’ll be doing when designing robots to solve a problem for customers, is going to require you to invent things that work under your current constraints of budget and operations. It’s possible that the robot you’re designing has never existed before.

Also you’ll need to choose the right software packages for your tasks, experiment with them, and decide what works best for your particular problem.

Therefore, having research experience is useful when entering the workforce and helps you throughout your career!

### Research Experience
Having a PhD in robotics is not necessary most of the time to work in the field, and it’s mostly valuable when you want to work as a research scientist in the industry.

However, keep in mind that even if you’re not working as a robotics researcher, following research breakthroughs in a continuous way is an important investment in your career and skills development.

Companies with big capital like Amazon for example have whole research divisions that hire top tier researchers to develop their algorithms and push their state of the art forward.

### Publications
Although having publications in general isn’t usually a hard requirement for robotics roles, some people do treat them as strong signals and data points in a person's portfolio.

The earlier you are in your career with publications, the more valuable it will serve your career. So if you can do some side research projects during your undergrad and publish your results in conferences or journals, it might help you a lot at the beginning of your career.

## Conclusion
Robotics is a field that, in my opinion, is still in its early stages of commercialization, with a lot of promise once Moore’s law hits the ratio of affordable computer power to cost.

The same way algorithms, compute, and data helped deep learning improve exponentially, robots have a huge potential to follow that curve when the time is ready.