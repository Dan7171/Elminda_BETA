# Investigating Advanced ML Algorithms for Determining the Efficacy of TMS Treatment in MDD Patients

![Project Logo](sagol_logo.jpg)

![image](https://github.com/Dan7171/Elminda_BETA/assets/103898362/8ce291cf-082a-4800-ac08-2bade30413fa)

## Introduction

This project is a brief description of your project and its purpose. It was developed as part of 0368352201 - Workshop on Computational Methods in Brain Research with the invaluable support and guidance from the following individuals:

- Professor Nitsan Censor from Tel Aviv University.
- Gil Issaschar from Firefly-Neuroscience Company.
- PHD Offir Laufer from Firefly-Neuroscience Company.


## Background and Importance:

Major Depressive Disorder (MDD) is a complex and prevalent mental health condition affecting millions of individuals worldwide. While traditional treatments, such as antidepressant medications and psychotherapy, have been the primary approach to managing MDD, Transcranial Magnetic Stimulation (TMS) has emerged as a promising alternative therapy. TMS is a non-invasive procedure that uses magnetic fields to stimulate specific regions of the brain associated with mood regulation.

Despite the increasing adoption of TMS in clinical settings, there is still a need to determine its efficacy accurately, particularly for individual patients. The response to TMS treatment can vary significantly from patient to patient, and predicting the likelihood of a positive outcome is challenging. Traditional diagnostic methods rely on subjective assessments, which can lead to delays in identifying the most suitable treatment option and potentially ineffective interventions.

## Project Overview

The primary objective of this research project is to investigate the application of advanced Machine Learning (ML) algorithms for predicting the efficacy of TMS treatment in MDD patients. By leveraging  datasets containing demographic, clinical, and neuroimaging information, we aim to develop predictive models that can accurately assess the likelih

## Usage and Permissions

1. Any usage, reproduction, or distribution of the code in this repository that falls under the terms of the agreement between the involved parties (e.g., Firefly-Neuroscience, TAU, and myself Dan Evron) is permitted and regulated by the terms of that agreement.

2. For any usage not explicitly covered by the aforementioned agreement, you must seek comprehensive permission from me, Dan Evron, the project owner, before using any code or files from this repository.

3. Unauthorized usage or distribution of the code without my explicit permission or violating the agreed terms with other parties will be considered a violation of rights and subject to legal action if necessary.

## Getting Started
Running locally:
Requirements: 1. python 3.8 and above 2. All installed packages 
Steps:
1. Open your project in pycharm or similar (important to select the repository and not a subdir)
2. Set your running configuration parameters for running in second_research\runArguments.py
3. Go to second_research\GSCVrunner_new.py and under next two if statements set your wanted param_pipe_list of the 
grid and pipe you want to run grid search on:


if args['classification']: 

    if args['lite_mode']:

       param_pipe_list = # set a list of param pipe list here
4. run second_research\GSCVrunner_new.py. This will trigger the cv grid search, find best
params for your pipeline and predict y values (responsive/non-responsive) on test set.
5. At second_research directory you'll see a new directory with current time. It contains graphs, results and important files describe your search train and prediction results

Running in cloud (google colab):
1. Upload the notebook grid_search_runner.ipynb to your colab user. It founds in this project's master branch.  
2. run steps 2 and 3 from previous section


## Contact

evrondan99@gmail.com
 
