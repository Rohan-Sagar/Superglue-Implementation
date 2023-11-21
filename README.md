# SuperGlue Reimplementation Project

## Overview
This repository contains my personal attempt at re-implementing the SuperGlue architecture, as described in the paper "SuperGlue: Learning Feature Matching with Graph Neural Networks" by Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich </br>
(Link: https://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf).</br>

This project is an educational endeavor to deepen my understanding of advanced feature-matching techniques in computer vision and graph neural networks.

SuperGlue represents a significant advancement in the field of computer vision, particularly in the area of feature matching. It leverages the power of Graph Neural Networks (GNNs) and attention mechanisms to establish correspondences between points in different images. This technology is pivotal in various applications, including 3D reconstruction, SLAM (Simultaneous Localization and Mapping), and augmented reality.

## Project Description
The goal of this project is to recreate the SuperGlue architecture from the ground up. This includes developing the attentional graph neural network, implementing the keypoint encoder, and utilizing the Sinkhorn algorithm for optimal matching. The project is structured to reflect the original paper's methodology while also incorporating my interpretations and learnings.

### Key Features
- Attentional Graph Neural Network implementation for understanding relationships between key points.
- Keypoint encoder to combine visual appearance and location information.
- Optimal matching layer with score prediction and the Sinkhorn algorithm.
- SLAM for SuperGlue algorithm
- SIFT descriptors with both brute-force matching and FLANN-based matching

## Credits and Acknowledgments
This reimplementation is inspired by and based on the research presented in the paper:

> Sarlin, P.-E., DeTone, D., Malisiewicz, T., & Rabinovich, A. (2020). SuperGlue: Learning Feature Matching with Graph Neural Networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

I extend my gratitude to the authors for their groundbreaking work in this field. This project is purely educational and not intended for commercial use.

## Installation and Usage
1) Install all the necessary packages mentioned in requirement.txt

2) Run the program using: './demo_superglue'

3) The flags used:
	a) --input -> location of the directory used for the input images
	b) --input/video_file_name -> location of the video used for running the model
	c) --output_dir -> directory where you want to store images of the files
	d) --use_sift -> evaluating the model on sift descriptor
	e) --use_sift --video -> evaluating the model on sift descriptor for videos and brute-force matching
	e) --use_sift --flann -> evaluating the model on sift descriptor with Flann-based matching

4) To measure the evaluation metrics on scanner dataset run ./match_pairs --eval



## License
As per the original paper license:</br>

SUPERGLUE: LEARNING FEATURE MATCHING WITH GRAPH NEURAL NETWORKS
SOFTWARE LICENSE AGREEMENT
ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution or non-profit organization or self (called "Licensee" or "You" in this Agreement) and Magic Leap, Inc. (called "Licensor" in this Agreement).  All rights not specifically granted to you in this Agreement are reserved for Licensor. 

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: 
Licensor retains exclusive ownership of any copy of the Software (as defined below) licensed under this Agreement and hereby grants to Licensee a personal, non-exclusive, non-transferable license to use the Software for noncommercial research purposes, without the right to sublicense, pursuant to the terms and conditions of this Agreement.  As used in this Agreement, the term "Software" means (i) the actual copy of all or any portion of code for program routines made accessible to Licensee by Licensor pursuant to this Agreement, inclusive of backups, updates, and/or merged copies permitted hereunder or subsequently supplied by Licensor,  including all or any file structures, programming instructions, user interfaces and screen formats and sequences as well as any and all documentation and instructions related to it, and (ii) all or any derivatives and/or modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY: Licensee acknowledges that the Software is proprietary to Licensor, and as such, Licensee agrees to receive all such materials in confidence and use the Software only in accordance with the terms of this Agreement.  Licensee agrees to use reasonable effort to protect the Software from unauthorized use, reproduction, distribution, or publication.

COPYRIGHT: The Software is owned by Licensor and is protected by United  States copyright laws and applicable international treaties and/or conventions.

PERMITTED USES:  The Software may be used for your own noncommercial internal research purposes. You understand and agree that Licensor is not obligated to implement any suggestions and/or feedback you might provide regarding the Software, but to the extent Licensor does so, you are not entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the Software, however, You agree that all and any such derivatives and modifications will be owned by Licensor and become a part of the Software licensed to You under this Agreement.  You may only use such derivatives and modifications for your own noncommercial internal research purposes, and you may not otherwise use, distribute or copy such derivatives and modifications in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies of the Software necessary for internal noncommercial use at a single site within its organization provided that all information appearing in or on the original labels, including the copyright and trademark notices are copied onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except as explicitly permitted herein. Licensee has not been granted any trademark license as part of this Agreement and may not use the name or mark "Magic Leap" or any renditions thereof without the prior written permission of Licensor.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in whole or in part, or provide third parties access to prior or present versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder without the prior written consent of Licensor. Any attempted assignment without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's acceptance of this Agreement by downloading the Software or by using the Software until terminated as provided below.

The Agreement automatically terminates without notice if you fail to comply with any provision of this Agreement.  Licensee may terminate this Agreement by ceasing using the Software.  Upon any termination of this Agreement, Licensee will delete any and all copies of the Software. You agree that all provisions which operate to protect the proprietary rights of Licensor shall remain in force should breach occur and that the obligation of confidentiality described in this Agreement is binding in perpetuity and, as such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this Agreement, there is no fee due to Licensor for Licensee's use of the Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON-INFRINGEMENT.  LICENSEE BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is provided as part of this Agreement.  

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent permitted under applicable law, Licensor shall not be liable for direct, indirect, special, incidental, or consequential damages or lost profits related to Licensee's use of and/or inability to use the Software, even if Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable U.S. export control laws, regulations, and/or other laws related to embargoes and sanction programs administered by the Office of Foreign Assets Control.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be invalid, illegal, or unenforceable by a court or other tribunal of competent jurisdiction, the validity, legality and enforceability of the remaining provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right or remedy under this Agreement shall be construed as a waiver of any future or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance with the laws of the State of Florida without reference to conflict of laws principles.  You consent to the personal jurisdiction of the courts of this County and waive their rights to venue outside of Broward County, Florida.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and entire agreement between Licensee and Licensor as to the matter set forth herein and supersedes any previous agreements, understandings, and arrangements between the parties relating hereto.
