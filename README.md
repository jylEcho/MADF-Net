\documentclass[journal,twoside,web]{ieeecolor}
\usepackage{tmi}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{float}


 
\usepackage[backend=biber,style=numeric,sorting=none]{biblatex}
\addbibresource{reference.bib} 


\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
\markboth{\journalname, VOL. XX, NO. XX, XXXX 2020}
{Author \MakeLowercase{\textit{et al.}}: Preparation of Papers for IEEE TRANSACTIONS ON MEDICAL IMAGING}


\begin{document}
\title{MADF-Net: Multi-phase Attentional Deep Fusion Network for Liver Tumor Segmentation}
\author{First A. Author, \IEEEmembership{Fellow, IEEE}, Second B. Author,and Third C. Author, Jr., \IEEEmembership{Member, IEEE}
\thanks{This paragraph of the first footnote will contain the date on which
you submitted your paper for review. It will also contain support information,
including sponsor and financial support acknowledgment. For example, 
``This work was supported in part by the U.S. Department of Commerce under Grant BS123456.'' }
\thanks{The next few paragraphs should contain the authors' current affiliations,
including current address and e-mail. For example, F. A. Author is with the
National Institute of Standards and Technology, Boulder, CO 80305 USA (e-mail:author@boulder.nist.gov). }
\thanks{S. B. Author, Jr., was with Rice University, Houston, TX 77005 USA.
He is now with the Department of Physics, Colorado State University,
Fort Collins, CO 80523 USA (e-mail: author@lamar.colostate.edu).}
\thanks{T. C. Author is with the Electrical Engineering Department,
University of Colorado, Boulder, CO 80309 USA, on leave from the National
Research Institute for Metals, Tsukuba, Japan (e-mail: author@nrim.go.jp).}}

\maketitle

\begin{abstract}
To address the limitations of existing multi-phase feature fusion in contrast-enhanced CT, we propose a Multi-phase Attention Deep Fusion Network (MADF-Net), which integrates arterial, portal venous, and delayed phase CT images through full-stage fusion at the input, feature, and decision levels. This design effectively exploits the complementary information among different phases. In addition, we introduce a boundary-aware dynamically weighted loss function, Boundary-enhanced Dynamic Loss (BED-Loss), to enhance the model’s ability to segment small lesions and indistinct boundaries. Experimental results on the LiTS2017 and MPLL datasets demonstrate that the proposed method outperforms existing approaches across multiple evaluation metrics, showing superior segmentation performance and strong potential for clinical application.
\end{abstract}

\begin{IEEEkeywords}
Liver Tumor Segmentation,Multi-phase CT,Self-Attention Mechanism,Deep Feature Fusion
\end{IEEEkeywords}

\section{Introduction}
\label{sec:introduction}
%\IEEEPARstart{L}{iver} tumor segmentation plays a vital role in quantitative medical image analysis. Accurate liver tumor segmentation provides essential morphological information that serves as a critical reference for clinical surgery and radiotherapy \cite{bilic2023liver}. Since the rise in popularity of deep learning, end-to-end frameworks based on fully convolutional neural networks, such as U-Net \cite{ronneberger2015u}, have been widely adopted. Variants of this architecture, like ResUNet and DenseNet \cite{diakogiannis2020resunet,huang2017densely} have achieved promising performance by integrating both 2D and 3D information to slice-level  or  volumetric-level features \cite{li2018h}. 
\IEEEPARstart{L}{iver} tumor segmentation is a critical task in quantitative medical image analysis, providing essential morphological and spatial information for surgical planning, radiotherapy, and post-treatment monitoring \cite{bilic2023liver}. With the advent of deep learning, fully convolutional neural networks (FCNs), particularly U-Net and its variants \cite{ronneberger2015u, diakogiannis2020resunet, huang2017densely}, have achieved notable success in automated segmentation tasks. These models extract features at either 2D slice-level or 3D volumetric-level, enabling robust representation learning for complex anatomical structures \cite{li2018h}.

% However, most of current liver tumor segmentation methods rely on single-phase CT images \cite{bilic2023liver,chen2021transunet,hatamizadeh2022unetr,cao2022swin}. Due to challenges such as low contrast and limited resolution, the segmentation accuracy of single-phase based networks often falls short of clinical requirements. Contrast-Enhanced Computed Tomography (CECT), which captures CT signals at different time points following contrast agent injection, offers a promising alternative. Specifically, it provides noncontrast (NC) enhanced phase, arterial (ART) phase, portal venous (PV) phase, and delayed (DL) phase \cite{chi2013content}. The NC phase provides fundamental anatomical information such as the general morphology, location, and size of the liver and tumors. (NC phase is omitted because of its weak complementary information and inability to reveal critical characteristics such as tumor vascularity and enhancement patterns \cite{ni2024tran,xu2021pa,liu2024pa}). The ART phase contains rich edge information of hyper-perfused regions, which facilitates tumor contour extraction and edge-enhanced feature representation \cite{kulkarni2021ct,urban2000helical}. The PV phase offers a comprehensive anatomical context and clearer liver parenchymal texture, ensuring structural completeness and providing stable tissue density information for more accurate segmentation \cite{kulkarni2021ct,schneider2014patient}. The DL phase is suitable for identifying fibrotic or hypo-perfused tumors, capturing residual features and tissue washout characteristics at later time points, thus complementing the information from the arterial and portal venous phases \cite{monzawa2007dynamic,lim2002detection}. Therefore, how to effectively extract and fuse the features from different phases has attracted the attention of many researchers.

However, the majority of existing methods rely solely on single-phase Computed Tomography (CT) images \cite{bilic2023liver, chen2021transunet, hatamizadeh2022unetr, cao2022swin}, often ignoring the phase-specific characteristics inherent in clinical imaging protocols. Due to low tissue contrast and resolution limitations, single-phase methods frequently fall short in achieving the precision required for clinical deployment. Contrast-Enhanced CT (CECT), which captures dynamic changes in tissue attenuation following contrast agent administration, provides a valuable alternative by acquiring images at multiple time points—typically including the non-contrast (NC), arterial (ART), portal venous (PV), and delayed (DL) phases \cite{chi2013content}.

Among these, the NC phase offers baseline anatomical information but lacks enhancement patterns relevant to tumor vasculature and lesion contrast, and is therefore generally not emphasized in liver tumor segmentation studies \cite{ni2024tran, xu2021pa, liu2024pa}. The ART phase captures early vascular features, highlighting hyper-perfused regions and enhancing lesion boundary delineation \cite{kulkarni2021ct, urban2000helical}. The PV phase provides clearer liver parenchyma and structural completeness, facilitating more accurate segmentation \cite{kulkarni2021ct, schneider2014patient}. The DL phase captures delayed enhancement and washout phenomena, aiding in the identification of fibrotic or hypo-perfused tumors \cite{monzawa2007dynamic, lim2002detection}. 
The complementary nature of these phases offers a compelling opportunity for improved segmentation through multi-phase fusion. Therefore, how to effectively extract and fuse the features from different phases has attracted the attention of many researchers.


% Existing feature fusion strategies can be categorized into three types \cite{zhang2021multi}: (1) Input-level fusion \cite{ouhmich2019liver}, where multi-phase CT images are directly concatenated before fed into the network and processed through a single network stream. (2) Feature-level fusion \cite{,xu2021pa,zhang2021multi,zhu2022medical,zhang2023multi,hazirbas2016fusenet}, where images from each phase are processed independently through separate encoders, and their features are subsequently fused and passed into a shared decoder. (3) Decision-level fusion \cite{sun2017automatic,raju2020co}, where each phase is processed by an independent encoder-decoder network, and fusion is performed at the final decision stage. Motivated by the need for effective multi-phase information integration, several studies have explored inter-phase feature extraction and fusion methods \cite{xu2021pa,zhang2021multi}. 

% The multi-phase fusion methods have demonstrated promising segmentation performance by integrating information from different contrast-enhanced CT phases, highlighting both the research value and clinical significance of multi-phase CT fusion in liver tumor segmentation.
% Existing input-level fusion methods struggle to capture complex nonlinear relationships \cite{zhang2023multi,sun2017automatic}. Feature-level fusion often fails to effectively handle ambiguous boundaries or low-confidence regions in the segmentation results \cite{sun2017automatic}. Decision-level fusion is limited by the frequent absence of modality data in clinical scenarios \cite{xu2021pa,zhu2022medical}. In addition, most existing approaches treat all CT phases as equally important during the fusion process, without fully exploring the complementary information among them. The unique characteristics and relative importance of each phase were overlooked \cite{xu2021pa,zhong2024prediction,qiao2024four}. Thus, How to effectivelly fuse multi-phase CT information to utilize their pros and avoid their cons remains an urgent challenge \cite{jiang2020multi}. In addition, due to the blurred lesion boundaries and low contrast commonly observed in multi-phase images, these methods also struggle to accurately model boundary regions. 





Existing multi-phase fusion strategies can be broadly categorized into three types \cite{zhang2021multi}: (1) \textit{Input-level fusion} \cite{ouhmich2019liver}, where multiple phases are concatenated as input and processed via a shared encoder; (2) \textit{Feature-level fusion} \cite{xu2021pa, zhang2021multi, zhu2022medical, zhang2023multi, hazirbas2016fusenet}, which extracts features from each phase independently before combining them at intermediate layers; and (3) \textit{Decision-level fusion} \cite{sun2017automatic, raju2020co}, where each phase is processed by a separate network and results are fused at the output level. 
While these approaches have shown potential, they often suffer from limitations such as insufficient modeling of nonlinear inter-phase relationships \cite{sun2017automatic, zhang2023multi}, reduced reliability in ambiguous or low-contrast regions, and vulnerability to missing-phase scenarios common in clinical workflows \cite{xu2021pa, zhu2022medical}. 
Moreover, many existing methods treat all phases with equal importance during fusion, overlooking their distinct clinical value and the complementary information they offer \cite{xu2021pa, zhong2024prediction, qiao2024four}. This results in suboptimal performance, especially in cases with blurred lesion boundaries or small lesions. Therefore, how to effectively fuse multi-phase CT features while leveraging their individual strengths and mitigating their limitations remains an open challenge \cite{jiang2020multi}.


%In this study, we at first quantitatively analyze the clinical contribution of the three enhanced CT phases separately and find that the PV phase yields the best segmentation performance. Based on this insight, we propose a novel attention-based deep fusion network, MADF-Net (Multi-phase Attention Deep Fusion Network), to effectively exploit complementary information across all three CT phases for improved liver tumor segmentation. MADF-Net performs full-stage deep fusion at the input, feature, and decision levels, enabling comprehensive inter-phase information interaction. Furthermore, to address the challenges of blurred boundaries and small lesion segmentation, we design a Boundary-enhanced Dynamic Loss (BED-Loss), which dynamically adjusts loss components according to lesion characteristics. This loss function enables adaptive coarse-to-fine segmentation and enhances the model’s boundary sensitivity in clinical scenarios, effectively mitigating the class imbalance issue and boundary inaccuracies.

In this paper, we begin by systematically evaluating the segmentation performance of each enhanced CT phase using standard deep learning models. Our quantitative analysis reveals that the PV phase contributes most significantly to segmentation accuracy, consistent with its known clinical role. Guided by this observation, we propose a novel framework, \textbf{\textit{MADF-Net}} (Multi-phase Attention Deep Fusion Network), to exploit the complementary advantages of the ART, PV, and DL phases through hierarchical fusion. MADF-Net introduces full-stage attention-based fusion across the input, feature, and decision levels, enabling deep inter-phase information interaction. To further address the challenges posed by class imbalance, low-contrast regions, and fuzzy lesion boundaries, we design a novel loss function called Boundary-enhanced Dynamic Loss (BED-Loss). This loss dynamically adjusts the emphasis on regional and boundary components based on lesion characteristics, facilitating adaptive coarse-to-fine segmentation and enhancing boundary sensitivity.
Extensive experiments on the public LiTS2017 dataset and a clinically collected multi-phase dataset (MPLL) demonstrate that our method achieves state-of-the-art segmentation performance, reaching a Dice score of 80.99\% when using all three phases, representing a 2.71\% improvement over the best single-phase baseline, and delivering consistent gains across all evaluation metrics on LiTS2017, confirming its robustness and generalizability.


% The main contributions of this work are as follows:
% \begin{enumerate}
% \item We quantitatively analyze the liver tumor segmentation performance of each CT phase and verify the clinical value of the PV phase in liver tumor segmentation.
% \item We propose MADF-Net, a self-attention-based deep fusion network designed for multi-phase liver tumor segmentation.
% \item We introduce a novel dynamically weighted loss function, BED-Loss, which integrates regional and boundary information to achieve more accurate contour prediction.
% \item The combination of proposed MADF-Net and BED-Loss achieve state-of-the-art (SOTA) liver tumore segmentation performance on two benchmark datasets: LiTS2017 and MPLL.
% \end{enumerate}

The main contributions of this study are summarized as follows:
\begin{enumerate}
\item We conduct a comprehensive quantitative analysis of liver tumor segmentation across different CT phases and demonstrate the predominant contribution of the portal venous (PV) phase, providing both empirical and clinical insights.
\item We propose MADF-Net, a multi-phase attention-based fusion network that integrates ART, PV, and DL phase features at multiple stages, enhancing liver tumor segmentation performance through deep inter-phase feature interaction.
\item We introduce a novel dynamically weighted loss function, BED-Loss, which integrates regional and boundary information to achieve more accurate contour prediction.
\item Extensive experiments on two benchmark datasets (LiTS2017 and MPLL) demonstrate that the proposed method achieves state-of-the-art liver tumor segmentation performance.
\end{enumerate}






\section{Related work}
\subsection{Single-Phase based liver tumore segmentation}
Deep learning has revolutionized single-phase liver tumor segmentation in CT images, Ronneberger et al. \cite{ronneberger2015u} proposed the U-Net architecture, with its encoder-decoder design and skip connections, remains foundational for single-phase liver tumor segmentation. Its ability to capture both local details and global context has been pivotal for handling the low contrast and fuzzy boundaries common in CT images. Similarly, H-DenseUNet \cite{li2018h} employs hybrid dense connections to enhance feature reuse, achieving state-of-the-art performance on the LiTS dataset for single-phase segmentation. Subsequent variants have optimized U-Net for efficiency and accuracy. UNet++ \cite{zhou2018unet++} uses nested skip connections to refine segmentation at multiple scales, addressing the challenge of small tumor detection in single-phase CT. These CNN-based models have set benchmarks in single-phase segmentation but often struggle with global context and long-range dependencies in complex tumor architectures. TransUNet \cite{chen2021transunet} pioneers a hybrid approach: a CNN encoder extracts low-level features from single-phase CT, while a Transformer encoder models global interactions between tumor and surrounding tissues. UNETR \cite{hatamizadeh2022unetr} and UNETR++ \cite{shaker2024unetr++}, combine Transformer’s global context with CNN’s local detail via a U-shaped encoder-decoder, achieving superior performance in single-phase 3D CT segmentation tasks.

\subsection{Multi-Phase based liver tumore segmentation}

Currently, more and more publications explore how to utilize the information of multi-phase CT scans to achieve better segmentation performance.  Multiphase fusion for liver tumor segmentation generally occurs in one of the three stages: input-level fusion, feature-level fusion, and decision-level fusion \cite{zhang2021multi}, which is called single-stage fusion in this paper. However, the fusion operation can also happen in more than one of these stages, which is called multi-stage fusion in this paper.

\subsubsection{Single-stage fusion}
An early example of input-level fusion was proposed by Ouhmich et al. in 2019 \cite{ouhmich2019liver}. Their method concatenated PV and ART phase images at the input stage and then fed them into a U-Net for training. The results showed that tumor segmentation metrics significantly outperformed those from training with single-phase images. Feature-level fusion of multi-phase CT data is currently the most active research area. Zhou et al. \cite{zhou2019hyper} first proposed a dual-path 3D fully convolutional network for pancreas segmentation, introducing skip connections across phases to enable dense information exchange. Wu et al. \cite{wu2019hepatic} modeled non-contrast and enhanced CT equivalently and applied combinatorial rules at specific U-Net layers for feature-level fusion. Decision-level fusion strategy trains the network for each period to extract the features of each period independently, and finally realizes feature fusion in the high-level layer. Raju et al. \cite{raju2020co} proposed a strategy of integrated joint training and semi-supervised training, which can effectively use a small amount of plain scan and enhanced CT data to robustly segment test data from different sources.

\subsubsection{Multi-stage fusion}
The fusion combination, especially the combination at feature-level and decision-level, is the main mode of multi-stage fusion network design at present \cite{ni2024tran,xu2021pa,liu2024pa,zhang2021multi,zhu2022medical,zhang2023multi,kuang2024adaptive}.

PA-Reseg \cite{xu2021pa} incorporates intra-phase and inter-phase attention mechanisms to capture channel-wise self-dependencies and cross-phase interactions. Attention modules were embedded at each encoder layer to fuse multi-scale information from ART and PV phases. Inspired by PA-ReSseg, SA-Net \cite{zhang2021multi} introduced a spatial aggregation module to ensure interaction during encoding, and used an uncertainty correction module at the decision stage to refine fuzzy tumor boundaries. To mitigate spatial misalignment in multi-phase medical images, Zhang et al. \cite{zhang2023multi} incorporated differentiable deformation operations \cite{jaderberg2015spatial} based on the previous models to align enhanced CT features. Ashwin et al. \cite{raju2020co} introduced a combined joint training and semi-supervised training strategy that effectively leverages limited non-contrast and enhanced CT data for robust segmentation across heterogeneous test datasets. However, this strategy is time-consuming due to multi-stage training. HRadNet \cite{liang2023hradnet} designed a multi-level feature extraction module based on a feature pyramid structure and introduced a metadata fusion layer to integrate clinical features such as tumor size and patient age, enhancing network generalization and adaptability to heterogeneous data. To further address spatial misalignment among multimodal images and boundary uncertainties, Zhang et al. \cite{zhang2023multi} proposed a multimodal tumor segmentation network that integrates a Deformable Aggregation Module (DAM) and an Uncertain Region Inpainting Module (URIM). By jointly utilizing differentiable deformable alignment and pixel-level aggregation strategies, this approach effectively mitigates registration errors caused by respiratory motion or scanning parameter variations across multi-phase CT.


\section{METHODOLOGY}
In this paper, we propose  MADF-Net to fuse the CT of all the three phases (ART, PV and DL) for liver tumor segmentation at the input, feature and decision levels. This section first introduces preliminary knowledge on fusion strategies, followed by the proposed MADF-Net, and then describes the designed BED-Loss.
\subsection{Preliminary}
As shown in Fig. 1, three common fusion strategies are illustrated.

\begin{figure}
\centering
\includegraphics[width=0.5\textwidth]{fusion_frameworkV10.0.png}
\caption{Multi-phase fusion method of enhanced CT. Fig (a), (b) and (c) show the input-level fusion, feature-level fusion and decision-level fusion architectures. Respectively.}
\label{fusion_frameworkV10.0.png}
\end{figure}

\paragraph{Input-level Fusion}
As illustrated in Fig. 1 (a), this strategy concatenates images from different phases (Phase 1, Phase 2, …, Phase n) along the channel dimension at the input stage to form a unified input tensor. Its mathematical expression is as follows:

\begin{equation}
I_{\text{input}} = \left[ I_1, I_2, \ldots , I_n \right] \in \mathbb{R}^{H \times W  \times nC},
\end{equation}
where \({{\rm{I}}_{\rm{1}}}{\rm{, }}{{\rm{I}}_{\rm{2}}}{\rm{, }}...{\rm{ , }}{{\rm{I}}_{\rm{n}}}\) denotes the image from the i-th phase (Phase 1, Phase 2, … , Phase n). After concatenating through the channel dimension, it forms \({I_{input}} \in {R^{(H \times W \times nC)}}\), where $H$, $W$ represent the height and width of the image. $C$ is the number of channels in a single-phase image ($C = 1$ in this paper).

\paragraph{Feature-level Fusion}
As shown in Fig. 1 (b), this strategy enhances the model's ability to capture complementary information across different phases by integrating multi-phase features through operations such as addition or concatenation. Let the feature maps extracted from each phase be \(I_1, I_2, \ldots, I_n\), then the fusion process can be expressed as:
\begin{equation}
I_{\text{fusion}} = \bigoplus_{i=1}^{n} I_i,
\end{equation}
where \(\bigoplus\) denotes fusion operations such as concatenation or addition. The fused feature map \(I_{fusion}\) is then fed into the decoder for segmentation.

\paragraph{Decision-level Fusion}
As shown in Fig. 1 (c), this strategy constructs separate segmentation branches for each phase. Features are extracted through their respective networks, the predictions from each phase are fused at the decision level to obtain the final output. Let the segmentation result of each phase be \({{\rm{S}}_{\rm{1}}}{\rm{, }}{{\rm{S}}_{\rm{2}}}{\rm{ , }}...{\rm{ , }}{{\rm{S}}_{\rm{n}}}\), then the final output \({{\rm{S}}_{final}}\) can be represented by the decision fusion mechanism as:
\begin{equation}{S_{final}}\; = {F}({S_1},{S_2}, \ldots ,{S_n}\;),\end{equation}
\noindent
where F denotes fusion methods such as averaging or weighted aggregation.

\subsection{MADF-Net architecture}

\begin{figure*}
\centering
\includegraphics[width=\textwidth]{MADF-Net.png}
\caption{MADF-Net network architecture. The figure illustrates the network’s forward propagation and error computation processes, as well as the detailed tensor operations within the encoder-decoder module. It also presents the comprehensive multi-stage fusion strategy, including input-level, feature-level and decision-level fusion.}
\label{MADF-Net.png}
\end{figure*}

To achieve input-level, feature-level and decision-level fusions in one network, we propose MADF-Net (Fig. 2). The whole MADF-Net consists of four encoder blocks (green part) and four decoder blocks (yellow part). The whole MADF-Net can be regarded as two U-Nets shouder-by-shouder: one U-Net (main branch) is called fusion branch (orange flow) and another U-Net (auxiliary branch) is called non-fusion branch. The fusion branch starts from the concatenated CT scans, which achieves the input-level fusion. Then, the images were passed to a series of encoders and decoders while keep receiving features from the non-fusion branch (auxiliary branch) at each encoder block and decoder block. By this way, we achieve the feature-level fusion. The decision-level fusion was implementated by concatenating the results from the final decoders of main branch and auxiliary branch  followed by a convolution block to output the final segmentation results. Therefore, the MADF-Net combined input-level, feature-level and decision-level fusion into one architecture. The details of this architecture are as follows.

At the input level, CT images of three phases (ART, PV and DL) are concatenated along the channel dimension, which can be expressed as:
\begin{equation}{I_{FUSION}} = {\rm{Concate}}\left( {{I_{ART}},{I_{DL}},{I_{PV}}} \right).\end{equation}

The concatenated images were fed into two branches: main branch (fusion branch) and auxiliary branch (non-fusion branch). The enocoders and decoders of fusion branch is nomrl convolusion, while the non-fusion branch uses three seperate covolution paths to seperatelly extracte the feature of different phases. 


During the encoding, the three-phase data and the fusion channel are simultaneously fed into each encoder block. The spatial resolution of the feature tensors is reduced by a factor of two, leading the width of feature decreased from 256 to 128, 64, and 32 gradually after each encoder block, while the number of channels were increased by a factor of four, from 4 to 16, 64 and 256 gradually.

To utilize the complementary information of different phases. The features from auxiliary branch are re-weighted using a self-attention mechanism:
\begin{equation}{{\rm{F}}_{FUSION}}{\rm{ = }}{{\rm{W}}_{ART}}{\rm{*}}{{\rm{F}}_{{\rm{ART}}}} + {{\rm{W}}_{DL}}{\rm{*}}{{\rm{F}}_{DL}} + {{\rm{W}}_{PV}}{\rm{*}}{{\rm{F}}_{PV}},\end{equation}
\noindent
here, \({{\rm{F}}_{^{{\rm{FUSION}}}}}\) represents the feature extraction result from the fusion branch. \({{\rm{W}}_{ART}}, {{\rm{W}}_{PV}}\) and \({{\rm{W}}_{DL}}\) represent the weights corresponding to the ART, PV and DL phases. 

These weights are obtained by a shared key-query scheme on patches to learn rich spatial-channel feature representations, as expressed below:
\begin{equation}
\begin{array}{l}
\mathrm{W}_{\mathrm{ART}} = \mathrm{softmax} \left( \frac{\mathrm{Q}_{\mathrm{ART}}^{\mathrm{P}} \left( \mathrm{K}_{\mathrm{ART}}^{\mathrm{P}} \right)^{\mathrm{T}}}{\sqrt{\mathrm{d}_{\mathrm{k}}}} \right), \\[6pt]
\mathrm{W}_{\mathrm{PV}} = \mathrm{softmax} \left( \frac{\mathrm{Q}_{\mathrm{PV}}^{\mathrm{P}} \left( \mathrm{K}_{\mathrm{PV}}^{\mathrm{P}} \right)^{\mathrm{T}}}{\sqrt{\mathrm{d}_{\mathrm{k}}}} \right), \\[6pt]
\mathrm{W}_{\mathrm{DL}} = \mathrm{softmax} \left( \frac{\mathrm{Q}_{\mathrm{DL}}^{\mathrm{P}} \left( \mathrm{K}_{\mathrm{DL}}^{\mathrm{P}} \right)^{\mathrm{T}}}{\sqrt{\mathrm{d}_{\mathrm{k}}}} \right),
\end{array}
\end{equation}
where \({d_k}\) is the dimension of the key, \({\rm{W}_{ART}}, {\rm{W}_{PV}}\), and \({\rm{W}_{DL}}\) represents the weighting coefficients of the three phases. 

After the feature fusion, a convolution block (Convolutin, Batch Normalization and ReLU activation)is applied. 






% During the decoding part, the intermediate decoding results from all encoder blocks are further fused.[再丰富一下]









In the decoding stage, the decoder receives four-channel output tensors \({{\rm{\tilde F}}_{{\rm{ART}}}}{\rm{, }}{{\rm{\tilde F}}_{{\rm{DL}}}}{\rm{, }}{{\rm{F}}_{{\rm{PV}}}}{\rm{,}}{{\rm{F}}_{{\rm{FUSION}}}}\) from the corresponding encoder layer and the output tensor \({{\rm{O}}_{{\rm{ART}}}}{\rm{,}}{{\rm{O}}_{{\rm{DL}}}},{{\rm{O}}_{{\rm{PV}}}},{{\rm{O}}_{{\rm{FUSION}}}}\) from the previous decoder layer. These tensors are concatenated along the channel dimension to form the decoder input. Each channel then undergoes upsampling convolution and batch normalization to dynamically increase the feature resolution by a factor of two. The three single-phase feature tensors are concatenated again along the channel dimension into the fusion channel \({{\rm{O}}_{{\rm{FUSION}}}}{\rm{ = }}{{\rm{O}}_{{\rm{ART}}}} \oplus {{\rm{O}}_{{\rm{PV}}}} \oplus {{\rm{O}}_{{\rm{DL}}}} \oplus {{\rm{O}}_{{\rm{FUSION}}}}\), which is followed by convolution operations to achieve multi-source feature fusion. This process enhances the integration of complementary information across channels and improves the overall feature representation.






















At the final stage of the decoder, decision-level fusion is performed between the decoder output and the convolutional feature maps to recover spatial information and enhance feature representation. This fusion enables pixel-wise mask prediction, yielding the final output 



\begin{equation}{O^{FINAL}} = {F_d}({O^{ART}},{O^{DL}},{O^{PV}},{O^{FUSION}}),\end{equation}
here, \({O^{ART}},{O^{DL}},{O^{PV}}\) represent the outputs of the three individual phases. \({O^{FUSION}}\) represents the prediction from the fused branch, \({O^{FINAL}}\) is the final segmentation result, and \({F_d}\) indicates the fusion strategy used.


Through the design of the encoder-decoder architecture, information fusion is achieved at the input level, intermediate feature level and output level, thereby realizing full-stage deep fusion across the entire network pipeline.

The proposed method achieves a full-process, all-stage deep collaborative modeling of cross-phase information, covering early, middle and late stages.

\subsection{BED-Loss function}
Cross-Entropy Loss and Dice Loss are widely used in medical image segmentation. However, these losses are often limited by blurred lesion boundaries and low contrast in multi-phase images, resulting in weak boundary region modeling. Boundary loss \cite{kervadec2019boundary}, on the other hand, improves the model’s sensitivity to shape by measuring the spatial distance between the predicted results and the true boundaries. 

As shown in Fig. 3, the boundary loss measures the spatial distance between the predicted region and the true label boundary by calculating the distance map. The two predictions have equal false positive area, resulting in identical overlap-based loss values, but the boundary loss varies due to the differences in contour geometry. In clinical practice, liver tumor normally have smooth contour, which means that the error in the Fig. 3-left should be penalized more than the error in Fig. 3-right. This can be achieved by introducing boundary loss.
 
To utilize the advantages of the three losses, therefore, we propose a dynamically weighted loss function, Boundary-enhanced Dynamic Loss (BED-Loss), which adaptively balances these three losses to handle both coarse and fine-grained segmentation.

During training, BED-Loss dynamically adjusts the model’s focus, effectively capturing contour errors under class imbalance and boundary ambiguity scenarios, and optimizes the subsequent boundary refinement process. 

\begin{figure}[h]
\centering
\includegraphics[width=0.5\textwidth]{BED-Loss.png}
\caption{The impact of differences in predicted boundaries on boundary loss (the red boundary in the figure represents the predicted tumor area, the blue circle represents the actual tumor area, and the green number represents the closest distance from the pixel to the boundary and the boundary loss)}
\label{BED-Loss.png}
\end{figure}

The implementation details of boundary loss are as follows. At first, the distance from a predicted positive pixel to ground truth contour is calculated:
\begin{equation}
\phi_{\mathrm{G}}(x) = 
\begin{cases}
\quad\quad0\quad, & x \in \mathrm{G} \\
d(x, \partial \mathrm{G}), & x \in \mathrm{G},
\end{cases}
\end{equation}
here, G denotes the ground-truth label (foreground region), \(\partial \)G denotes the edge contour of G, \(d(x,\partial G)\) denotes the shortest distance from pixel x to the ground-truth boundary \(\partial G\), and \(  d(x,\partial G) = {\min _{y \in \partial G}}\left\| {x - y} \right\|\) represents the ground truth label, where y is any point on the boundary \(\partial G\), and \(\left\|  \cdot  \right\|\) indicates the Euclidean distance. Based on this, the Boundary Loss is defined as:
\begin{equation}{{\cal L}_{boundary}} = \int_\Omega  {{y_i} \cdot \left| {{\phi _G}(x)} \right|dx},\end{equation}
\noindent
here, \({{\rm{y}}_{\rm{i}}} \in [0,1]\) denotes the predicted probability that a pixel $x$ belongs to each class, and \(\Omega \) represents the image domain.

From the above definition we can see that boundary loss focus on the total shape nearness. Dice Loss maximizes the spatial consistency between the predicted and ground truth regions, while Cross-Entropy Loss enhances the model’s discriminative ability in homogeneous regions, ensuring accurate classification at the pixel level. 

To combine the advantages of these three losses, we propose BED-Loss to combine Cross-Entropy Loss, Dice Loss and Boundary Loss through dynamic weighting, as expressed below:
\begin{equation}Loss = \alpha  \cdot {L_{D{\rm{ice}}}} + \beta  \cdot {L_{CE}} + \gamma  \cdot {L_{BL}},\end{equation}
here, \(\alpha\), \(\beta \) and \(\gamma\) are the loss weighting coefficients.
\section{Experiments}
\subsection{dataset}
Two datasets were used in this study: the publicly available Liver Tumor Segmentation Benchmark (LiTS) dataset \cite{bilic2023liver} and an in-house multi-phase liver lesion CT (MPLL) dataset. LiTS contains single-phase CT scans, while MPLL consists of multi-phase scans.

\textbf{Single-phase dataset} The LiTS dataset consists of 201 abdominal PV phase CT scans collected from seven clinical institutions worldwide. Among these, 194 scans contain liver lesions. The number of tumors per case varies from 0 to 12, with tumor volumes ranging from 38 mm³ to 1231 mm³. In this study, we selected 131 CT scans from the official LiTS training set for experimentation.

\textbf{Multi-phase dataset} The MPLL dataset consists of 141 patients with liver disease from the First Affiliated Hospital of University of Science and Technology of China. The dataset includes patients aged between 9 and 72 years, and the number of axial slices per scan varying from 48 to 777. All the images in MPLL dataset contain three enhanced phases (ART, PV and DL). In this study, the B-spline registration algorithm was employed to register arterial phase and delayed phase images using portal venous phase images as the reference. The registered images were annotated using ITK-SNAP software by two experienced attending radiologists, and subsequently reviewed by a third attending radiologist to ensure the accuracy and consistency of the annotations. All data have been anonymized and contain only image information. The MPLL dataset has received approval from the institutional ethics committee.


Example images from both datasets are shown in Fig. 4. Both the LiTS2017 and MPLL datasets were split into training, validation and test sets with a 7:1:2 ratio \cite{jiang2023rmau}, as summarized in Table I.

\begin{figure}[h]
\centering
\includegraphics[width=0.45\textwidth]{Dataset.png}
\caption{Example images from the LiTS and MPLL datasets (red indicates liver regions, green indicates tumor regions).}
\label{Dataset.png}
\end{figure}


\begin{table}[h]
\centering
\caption{Dataset characteristics}
\begin{tabular}{ccc}
\hline
Dataset & Attribute & Value \\
\hline
         & Phase & PV \\
LiTS2017         & Slice thickness & 0.45mm--6.0mm \\
         & Slice resolution & 512 × 512 \\
         & Disease type & HCC, ICC, etc. \\
\hline
         & Phase & ART, PV, DL \\
MPLL     & Slice thickness & 0.62mm--5.0mm \\
         & Slice resolution & 512 × 512 \\
         & Disease type & ABS, HCC, HEM, ICC, Lipoma \\
\hline
\end{tabular}
\end{table}

\subsection{Evaluation Metrics}
We employed the Dice Similarity Coefficient (DSC), Jaccard Similarity Coefficient (JSC), Average Symmetric Surface Distance (ASSD), and 95\% Hausdorff Distance (HD$_{95}$) to evaluate the experimental results. These metrics are defined as follows (let $P$ denote the predicted segmentation and $G$ denote the ground truth label).

\noindent Dice Similarity Coefficient (DSC):
\begin{equation}DSC = \frac{{2 \times \left| {P \cap G} \right|}}{{\left| P \right| + \left| G \right|}},\end{equation}
\noindent
Jaccard Similarity Coefficient (Jaccard):
\begin{equation}J{\rm{accard}} = \frac{{\left| {P \cap G} \right|}}{{\left| {P \cup G} \right|}},\end{equation}
\noindent
95\% Hausdorff Distance (HD$_{95}$) \cite{jiang20253d}:
\begin{align}
HD_{95}(G, P) = \max \big\{ 
    &\max_{g \in G} \min_{p \in P} \left\| g - p \right\|, \notag\\
    &\max_{p \in P} \min_{g \in G} \left\| p - g \right\| 
\big\},
\end{align}
\noindent
Average Symmetric Surface Distance (ASSD):
\begin{equation}ASSD = \frac{{\sum\nolimits_{{\rm{p}} \in P} {dist(p,G) + } \sum\nolimits_{g \in G} {dist(g,P)} }}{{\left| P \right| + \left| G \right|}}.\end{equation}
\subsection{Implementation Details}
% The proposed method was implemented based on the Linux 5.4.0 operating system using PyTorch 1.13.1. All models were trained and evaluated on NVIDIA GeForce RTX 3090 GPUs (24 GB $\times$ 2). We adopted Stochastic Gradient Descent (SGD) as the optimizer, with a learning rate of 0.00011 and 4 data loading processes. The models were trained for a total of 450 epochs with a batch size of 8.

% During early training, the dynamic weighting strategy of BED-Loss, \(\alpha\), \(\beta \) are initialized at 0.49, and \(\gamma\) at 0.02. If the loss stagnates for 10 epochs, the weights are adjusted to a 4:4:2 ratio, promoting a shift from coarse to fine segmentation and enhancing boundary accuracy.

All models were trained for 450 epochs with a batch size of 8. The Stochastic Gradient Descent (SGD) optimizer was adopted with a learning rate of 0.00011 and 4 parallel data loading workers. These hyperparameters were selected to ensure stable convergence and efficient training across all experimental settings.

% In the early stages of training, the dynamic weighting strategy of BED-Loss is applied, where the initial weights of \(\alpha\) and \(\beta \) are set to 0.49, and \(\gamma\) is set to 0.02. When the overall loss remains stagnant for 10 consecutive epochs, the weight configuration is adjusted to a ratio of 4:4:2, enabling a progressive transition from coarse to fine segmentation. This strategy effectively improves the delineation of tumor boundaries and enhances segmentation accuracy.

The initial weights of BED-Loss in Eq. (10) are set to \(\alpha\)=0.49, \(\beta \)=0.49 and \(\gamma\)=0.02. If the training loss plateaus for 10 epochs, the weights are dynamically adjusted to a 4:4:2 ratio. 
% This strategy facilitates a gradual shift from coarse to fine segmentation, enhancing boundary localization accuracy.

The proposed method was implemented on a Linux 5.4.0 system using PyTorch 1.13.1. All experiments were conducted on two NVIDIA GeForce RTX 3090 GPUs (24 GB × 2), providing sufficient computational resources for efficient model training and evaluation.

\section{Experimental results}

We conducted a series of experiments to explore the performance of our method and compare our method with the existing methods. 

At first,  a group of single-phase (1P) experiments were conducted to compare the performance of different phases. Six existing methods \cite{chowdary2024med, wang2021attu, zheng2024assnet, chen2021transunet, valanarasu2021kiu, azad2023dae}, which were specifically designed for single-phase inputs and have achieved promising results, were applied on the same dataset to have a fair comparison with our method.

Then we conducted a group of two-phase (2P) and a group of three-phase (3P) experiments based on our MADF-Net and five existing multi-phase methods \cite{sun2017automatic, zhu2022medical, zhang2021multi, xu2021pa, kuang2024adaptive}.

After that, some ablation experiments were finished to explore the influence of the attention model of MADF-Net and our proposed BED-Loss.

As all of the aforementioned existing methods have publicly available source code, we conducted experiments on the MPLL and LiTS datasets using their official implementations. Since the in-house MPLL dataset contains multi-phase scans, we used it to conduct experiments using 1P, 2P and 3P inputs for a fair comparison. Although the MPLL dataset is suitable for evaluating the effectiveness of the proposed BED Loss, we further performed ablation studies on the LiTS dataset to demonstrate its generalizability across different datasets. 

\subsection{Main Result}
We conducted three groups of experiments to explore the optimal combination for 1P, 2P and 3P settings. The best combinatioin at each group is show at Table II. With the increase in the number of input phases, the segmentation accuracy improves accordingly, demonstrating the effectiveness of our deep fusion network in fully exploiting the complementary information across multiple phases. The details of these experiments are as follows.
\begin{table}[h]
\centering
\caption{Comparison of different phases based on various metrics.}
\label{tab:phase_comparison}
\begin{tabular}{lcccc}
\hline
Phase & DSC & Jaccard & HD$_{95}$ & ASSD \\
\hline
Single Phase (PV) & 78.28 & 0.6431 & 3.4972 & 4.37 \\
Two-Phase (PV+ART) & 78.56 & 0.6432 & 2.8569 & 7.56 \\
Three-Phase (ART+PV+DL)  & \textbf{80.99} & \textbf{0.6805} & \textbf{2.5948} & \textbf{4.26} \\
\hline
\end{tabular}
\end{table}

The three-phase combination (ART+PV+DL) achieved the highest DSC (80.99), Jaccard index (0.6805) and the lowest ASSD (4.26), HD$_{95}$ (2.5948), indicating superior overall segmentation accuracy and boundary consistency. These results demonstrate that incorporating all three phases leads to the most balanced and robust performance across all metrics.
\subsection{Experiments Result on 1P (ART, PV or DL)}

In this study, we conducted separate training and evaluation on images from the PV phase (MPLL-P), ART phase (MPLL-A) and DL phase (MPLL-D) of the MPLL dataset, using several state-of-the-art segmentation models. The results are presented in Table III.



\begin{table*}[htbp]
\centering
\caption{Quantitative comparison of different methods on the MPLL dataset.}
\begin{tabular}{l cccc cccc cccc}
\hline
\multicolumn{1}{c}{} & \multicolumn{4}{c}{MPLL-A} & \multicolumn{4}{c}{MPLL-P} & \multicolumn{4}{c}{MPLL-D} \\
\hline
Model & DSC & Jaccard & HD$_{95}$ & ASSD & DSC & Jaccard & HD$_{95}$ & ASSD & DSC & Jaccard & HD$_{95}$ & ASSD \\
\hline
MedFormer \cite{chowdary2024med}         
& 0.7251 & 0.5688 & 9.5347 & 5.8928 
& \textbf{0.7309} & \textbf{0.5759} & \textbf{9.0265} & \textbf{5.5638} 
& 0.7023 & 0.5412 & 13.7836 & 6.9864 \\
AttUNet \cite{wang2021attu}          
& 0.7266 & 0.5706 & 9.3260 & 5.2862 
& \textbf{0.7382} & \textbf{0.5850} & \textbf{0.8276} & \textbf{8.5263} 
& 0.7257 & 0.5695 & 9.6487 & 6.0707 \\
ASSNet \cite{zheng2024assnet}            
& 0.7630 & 0.6168 & 5.5263 & 4.2452 
& \textbf{0.7714} & \textbf{0.6279} & \textbf{3.3584} & \textbf{4.8079} 
& 0.7396 & 0.5868 & 8.0483 & 5.7039 \\
TransUNet \cite{chen2021transunet}          
& 0.7633 & 0.6172 & 5.6885 & 4.9167 
& \textbf{0.7662} & \textbf{0.6210} & \textbf{5.2648} & \textbf{4.4610} 
& 0.7416 & 0.5893 & 8.0719 & 5.7280 \\
KiU-Net \cite{valanarasu2021kiu}           
& 0.7526 & 0.6172 & 6.7645 & 4.6654 
& \textbf{0.7793} & \textbf{0.6384} & \textbf{3.9364} & \textbf{4.6094} 
& 0.7447 & 0.5932 & 7.9564 & 5.2831 \\
DAE-Former \cite{azad2023dae}        
& 0.7727 & 0.6296 & 4.7654 & 4.6685 
& \textbf{0.7828} & \textbf{0.6431} & \textbf{3.4972} & \textbf{4.3740} 
& 0.7594 & 0.6121 & 5.9371 & 5.2905 \\
\hline
\end{tabular}
\end{table*}


It was observed that all models achieved higher Dice Similarity Coefficient (DSC) and Jaccard Index on the portal venous phase (MPLL-P) compared to the arterial phase (MPLL-A) and delayed phase (MPLL-D), indicating better segmentation accuracy and greater overlap with the ground truth in MPLL-P. Furthermore, the models yielded the lowest or near-lowest HD$_{95}$ and ASSD on MPLL-P, suggesting that the segmentation boundaries generated in the portal venous phase are closer to the actual anatomical boundaries. 
In clinical practice, this may indicates that the portal venous phase provides more pronounced grayscale contrast between different tissues and between lesions and normal structures \cite{ni2024tran,lam2017value,al2024multi}. This is especially beneficial in cases where tumor boundaries are ambiguous and exhibit high heterogeneity \cite{liu2024pa}.

\subsection{Experiments Result on 2P (ART and PV)}

As shown in the 1P experimental results, the models trained on arterial or portal venous phases achieved better performance than those on DL phase. Based on this observation, we used artery and portal venous phases of MPLL dataset to conduct the 2P experiments. As shown in Table IV, MADF-Net achieved superior performance, improving the DSC metric by 7.48\%, 3.97\%, 2.11\%, 3.04\% and 1.18\% over the other five methods, respectively. It also consistently outperformed the others in terms of ASSD and HD$_{95}$. These findings demonstrate the effectiveness of the deep fusion architecture on the 2P fusion strategy.

\begin{table}[H]
\centering
\caption{Quantitative display of results on the MPLL-2P dataset. “Phase” indicates the data modality used. (The bold indicators indicate the best performance.)}
\begin{tabular}{ c c c c c c }
\hline
\textbf{Phase} & \textbf{Model} & \textbf{DSC} & \textbf{Jaccard} & \textbf{HD$_{95}$} & \textbf{ASSD} \\
\hline
A+P & MC-FCN \cite{sun2017automatic}          & 0.7108 & 0.5513 & 10.5972 & 11.65 \\
A+P & MW-UNet \cite{zhu2022medical} & 0.7459 & 0.5948 & 7.6918 & 8.59 \\
A+P & SA-Net \cite{zhang2021multi}   & 0.7645 & 0.6167 & 5.5816 & 12.15 \\
A+P & PA-ResSeg \cite{xu2021pa}  & 0.7552 & 0.6092 & 6.3594  & \textbf{6.21} \\
A+P & MCDA-Net \cite{kuang2024adaptive}  & 0.7738 & 0.6390 & 4.5264  & 10.78 \\
\hline
A+P &   \textbf{Ours}    & \textbf{0.7856} & \textbf{0.6432} & \textbf{2.8569} & 7.56 \\
\hline
\end{tabular}
\end{table}




\subsection{Experiments Result on 3P (ART, PV and DL)}


\begin{table}[htbp]
\centering
\caption{Quantitative display of results on the MPLL-3P dataset.}
\begin{tabular}{ c c c c c c }
\hline
\textbf{Phase} & \textbf{Model} & \textbf{DSC(\%)} & \textbf{Jaccard} & \textbf{HD$_{95}$} & \textbf{ASSD} \\
\hline
A+P+D & MC-FCN \cite{sun2017automatic}        & 72.85 & 0.5841 & 9.3652 & 6.02 \\
A+P+D & MW-UNet \cite{zhu2022medical} & 74.02 & 0.6031 & 7.0967 & 12.15 \\
A+P+D & SA-Net \cite{zhang2021multi}        & 75.29 & 0.6248 & 6.9182  & 5.10 \\
A+P+D & PA-ResSeg \cite{xu2021pa}     & 77.61 & 0.6356 & 4.5984  & 5.44 \\
A+P+D & MCDA-Net \cite{kuang2024adaptive} & 78.42 & 0.6458 & 3.2564  & 6.91 \\
\hline
A+P+D & \textbf{Ours} & \textbf{80.99} & \textbf{0.6805} & \textbf{2.5948} & \textbf{4.26} \\
\hline
\end{tabular}
\label{tab:comparison}
\end{table}

\begin{figure*}
\centering
\includegraphics[width=\textwidth]{cuttingV4.0.png}
\caption{Comparison of the three-phase network experiment results. For better visualization, we performed appropriate cropping. (The green region in the ground truth (GT) represents the tumor, the green region in the prediction indicates the predicted tumor area, and the red region denotes the difference between the two.)}
\label{cuttingV4.0.png}
\end{figure*}

The comparison between our MADF-Net and other existing five multi-phase methods on 3P fusion strategy are shown in Table V. Compared to other existing methods, the proposed MADF-Net achieved higher DSC and Jaccard, lower HD$_{95}$ and ASSD. That means that our MADF-Net more accurately  localizing and delineating the spatial positions and geometric shapes of the target regions.

\begin{figure*}
\centering
\includegraphics[width=\textwidth]{三期比较V9.0.png}
\caption{Comparison of the three-phase network experiment results, the figure illustrates the comparison of segmentation performance on the MPLL-1P, MPLL-2P, and MPLL-3P datasets, evaluated using multiple quantitative metrics including DSC, VOE, RAVD, and ASSD.}
\label{三期比较V9.0.png}
\end{figure*}

As shown in Fig. 6, the proposed deep fusion network, MADF-Net, outperforms mainstream methods such as MW-UNet in terms of the DSC metric. It also achieves lower values in VOE, RAVD and ASSD, highlighting its advantage in accurately segmenting multi-phase data. This improvement is primarily attributed to the deep fusion strategy and the effective utilization of information from all three phases.






\section{Analysis And Discussion}
\subsection{Abalition study on MPLL-3P}

Table VI show the ablation study to explore the influence of self-attention model and BED-Loss. The results of f and g verified that both the inter-channel self-attention (SA) and the BED-Loss (BED) contributed positively to segmentation performance, with DSC improvements of 2.00\% and 1.63\%, and HD$_{95}$ reductions of 2.2693 and 1.2700, respectively. These results demonstrate that the lossless fusion strategy employed in MADF-Net can effectively refine and integrate multi-phase features in most scenarios. When both components were applied simultaneously, the DSC increased by 2.47\% and HD$_{95}$ decreased by 3.2693, confirming that the fusion and deep fusion networks can make full use of diverse domain information and assign distinct weights to features from different phases, thereby enhancing the overall feature representation capability.

\begin{table}[H]
\centering
\caption{Ablation study results of MADF-Net on MPLL-3P}
\begin{tabular}{ c c c c c c c }
\hline
\textbf{Version} & \multicolumn{2}{c}{\textbf{Experiments}} & \multicolumn{4}{c}{\textbf{MADF-Net}} \\
\cline{2-7}
 & SA&BED&DSC(\%)&Jaccard&HD$_{95}$&ASSD \\
\hline
e &              &              & 78.52 & 0.6685 & 5.8641 & 7.63 \\
f &              & \checkmark   & 80.15 & 0.6952 & 4.5941 & 4.25 \\
g & \checkmark   &              & 80.52 & 0.7059 & 3.5948 & 5.26 \\
h & \checkmark   & \checkmark   & \textbf{80.99} & \textbf{0.6805} & \textbf{2.5948} & \textbf{4.26} \\
\hline
\end{tabular}
\label{tab:ablation}
\end{table}



\subsection{Analysis of BED-Loss Function}

Table VII and Fig. 7 (b) presents a quantitative and qualitative comparison of the BED-Loss applied to mainstream liver tumor segmentation methods. 

\begin{figure}
\centering
\includegraphics[width=0.5\textwidth]{modifyV4.0.png}
\caption{Experimental results on the LiTS2017 dataset. (Fig. 7(a) shows the annotated liver tumors along with their corresponding distance maps, the gray area in the mask represents the liver, while the white area indicates the tumor. Fig. 7(b) presents the segmentation results of two LiTS2017 cases using six different segmentation methods, the green area denotes the predicted liver region and the red area indicates the predicted tumor region.)}
\label{modifyV4.0.png}
\end{figure}

\begin{table}[h]
\centering
\caption{Quantitative comparison of contrast methods on the LiTS dataset (bold indicates training using BED-Loss.) }
\begin{tabular}{ l c c c c }
\hline
\textbf{Models} & \textbf{DSC(\%)} & \textbf{Jaccard(\%)} & \textbf{HD$_{95}$} & \textbf{ASSD} \\
\hline
MedFormer \cite{chowdary2024med} & 81.55 & 70.75 & 8.4725 & 2.3184 \\
\textbf{+BED-Loss} & \textbf{82.03}  & \textbf{73.12}  & \textbf{6.5478}  & \textbf{1.8847}  \\
\hline
AttUNet \cite{wang2021attu} & 75.8 & 72.45 & 9.8641 & 2.9675 \\
\textbf{+BED-Loss} & \textbf{76.39}  & \textbf{74.52} & \textbf{7.1216}  & \textbf{2.3368}  \\
\hline
ASSNet \cite{zheng2024assnet} & 77.9 & 73.09 & 8.7312 & 2.6843 \\
\textbf{+BED-Loss} & \textbf{78.33} & \textbf{75.22}  & \textbf{6.1329}  & \textbf{2.0945}  \\
\hline
TransUNet \cite{chen2021transunet} & 82.15 & 79.85 & 4.3827 & 1.1388 \\
\textbf{+BED-Loss} & \textbf{82.77}  & \textbf{81.92}  & \textbf{2.6938}  & \textbf{0.8743}  \\
\hline
KiU-Net \cite{valanarasu2021kiu} & 81.35 & 75.65 & 6.4783 & 1.9372 \\
\textbf{+BED-Loss} & \textbf{81.89}  & \textbf{77.48}  & \textbf{4.2793}  & \textbf{1.3428}  \\
\hline
DAE-Former \cite{azad2023dae} & 83.75 & 77.7 & 4.9156 & 1.4721 \\
\textbf{+BED-Loss} & \textbf{84.19}  & \textbf{80.68}  & \textbf{2.9437}  & \textbf{1.0846}  \\
\hline
\end{tabular}
\label{tab:bed_loss_comparison}
\end{table}


All models trained with the BED-Loss showed improvements compared to their respective baselines, indicating that the BED-Loss helps the models better capture the similarity between the liver tumor regions and the ground truth. The models also exhibited varying degrees of improvement in HD$_{95}$ and ASSD metrics. For example, the HD$_{95}$ of TransUNet decreased from 4.3827 to 2.6938, and the ASSD reduced from 1.1388 to 0.8743.

In summary, the BED-Loss effectively optimizes the performance of mainstream segmentation models on key metrics such as DSC, Jaccard, HD$_{95}$, and ASSD, thereby enhancing the segmentation accuracy and boundary precision of liver tumors.



\section{Conclusion}

MADF-Net, by effectively integrating complementary information from different imaging phases, achieved more accurate boundary delineation, as supported by its superior HD$_{95}$ and ASSD values. This improvement may be attributed to its deeper fusion mechanism, which goes beyond simple channel-wise convolution and inter-phase attention to capture more comprehensive contextual information in complex medical images.

\printbibliography

\end{document}
