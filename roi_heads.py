from __future__ import division
import torch
import torchvision
from torch.autograd import Variable

import torch.nn.functional as F
from torch import nn, Tensor

from torchvision.ops import boxes as box_ops
from torchvision.ops import misc as misc_nn_ops

from torchvision.ops import roi_align

from torchvision.models.detection import _utils as det_utils

from torch.jit.annotations import Optional, List, Dict, Tuple

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def mutual_loss(class_logits, box_regression, labels, regression_targets, weirules_model=None, deep_inp_in=None, rules_inp=None):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor])
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """
    classification_loss_w = None
    
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    if weirules_model==None:
        classification_loss_f = F.cross_entropy(class_logits, labels)
    else:
        
        # WEIRULES cross entropy
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset].clone().detach()
        deep_inp=deep_inp_in[sampled_pos_inds_subset].clone().detach()
        rules_inp=rules_inp[sampled_pos_inds_subset].clone().detach()      
        wei_logits = weirules_model.forward_model(deep_inp, rules_inp)
        
        tensor_class_weights=torch.tensor(weirules_model.class_weights).float().cuda()
        lw =  F.cross_entropy(wei_logits, labels_pos-1,tensor_class_weights)
        
        # FRCNN cross entropy
        lf =  F.cross_entropy(class_logits, labels)
        
        # KL div frcnn - weirules
        class_logits_pos=class_logits[sampled_pos_inds_subset]
        
        divf = F.kl_div(F.log_softmax(class_logits_pos[:,1:], dim=1), F.softmax(Variable(wei_logits), dim=1),reduction='batchmean')
        divw = F.kl_div(F.log_softmax(wei_logits, dim=1), F.softmax(Variable(class_logits_pos[:,1:]), dim=1),reduction='batchmean')
        

        classification_loss_f = lf + divf
        classification_loss_w = lw + divw

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss_f, box_loss, classification_loss_w


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor])
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss, 0


class RoIHeads(torch.nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img
                 ):
        super(RoIHeads, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
   
    
    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor])
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
            match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
            matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

            clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

            labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
            labels_in_image = labels_in_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_in_image[bg_inds] = torch.tensor(0)

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = torch.tensor(-1)  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor])
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor])
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def DELTEME_all(self, the_list):
        # type: (List[bool])
        for i in the_list:
            if not i:
                return False
        return True

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]])
        assert targets is not None
        assert self.DELTEME_all(["boxes" in t for t in targets])
        assert self.DELTEME_all(["labels" in t for t in targets])
      
    def select_training_samples(self, proposals, targets, rule_input):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        matched_rule_inputs = []
        num_images = len(proposals)
            
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            matched_gt_boxes.append(gt_boxes[img_id][matched_idxs[img_id]])
        
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)

        if rule_input is not None:
            oversampled_rule_input=[]
            for img_id in range(num_images):
                rule_input_per_image=[]
                for idx in matched_idxs[img_id]:
                    rule_input_per_image.append(rule_input[img_id][idx])
                rule_input_per_image=torch.stack(rule_input_per_image, dim=0)
                oversampled_rule_input.append(rule_input_per_image)
            oversampled_rule_input = torch.cat(oversampled_rule_input, dim=0)
            return proposals, matched_idxs, labels, regression_targets, oversampled_rule_input

        return proposals, matched_idxs, labels, regression_targets, None
    
    
    def postprocess_detections_extract_df(self, class_logits, box_regression, proposals, image_shapes, deep_inp_in):
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)
 
        # split boxes and scores per image
        if len(boxes_per_image) == 1:
            # TODO : remove this when ONNX support dynamic split sizes
            # and just assign to pred_boxes instead of pred_boxes_list
            pred_boxes_list = [pred_boxes]
            pred_scores_list = [pred_scores]
            deep_inp_list = [deep_inp_in]
        else:
            pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
            pred_scores_list = pred_scores.split(boxes_per_image, 0)
            deep_inp_list = deep_inp_in.split(boxes_per_image, 0)

        all_predictions_di=[]
        for boxes, scores, image_shape, deep_inp in zip(pred_boxes_list, pred_scores_list, image_shapes, deep_inp_list):
            bboxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            
            
            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            
            num_to_m = boxes.shape[1]
            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            deep_inp = torch.cat([deep_inp for _ in range(num_to_m)],dim=0)
            # remove low scoring boxes
            inds = torch.nonzero(scores > 0).squeeze(1)
            boxes, scores, labels, deep_inp = boxes[inds], scores[inds], labels[inds], deep_inp[inds]
            
            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, deep_inp = boxes[keep], scores[keep], labels[keep], deep_inp[keep]
            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only top1 scoring predictions
            keep = keep[:1]
            boxes, scores, labels, deep_inp = boxes[keep], scores[keep], labels[keep], deep_inp[keep]

            all_predictions_di.append(deep_inp)
        return all_predictions_di
           
    def postprocess_detections_com(self, class_logits, box_regression, proposals, image_shapes, weirules_model=None, deep_inp=None, rules_inp=None):
        # type: (Tensor, Tensor, List[Tensor], List[Tuple[int, int]], weirules, Tensor, Tensor)
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        
        pred_scores = F.softmax(class_logits, -1)

   
        if weirules_model!=None:
            wei_logits = weirules_model.forward_model(deep_inp, rules_inp)
            #wei_pred_scores = F.normalize(wei_logits, p=1, dim=1)
            wei_pred_scores = F.softmax(wei_logits, dim=1)


        # split boxes and scores per image
        if len(boxes_per_image) == 1:
            # TODO : remove this when ONNX support dynamic split sizes
            # and just assign to pred_boxes instead of pred_boxes_list
            pred_boxes_list = [pred_boxes]
            pred_scores_list = [pred_scores]
            wei_pred_scores_list = [wei_pred_scores]
        else:
            pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
            pred_scores_list = pred_scores.split(boxes_per_image, 0)
            wei_pred_scores_list = wei_pred_scores.split(boxes_per_image, 0)


        all_boxes = []
        all_scores = []
        all_labels = []
        all_wscores = []
        all_wlabels = []
        for boxes, scores, wscores, image_shape in zip(pred_boxes_list, pred_scores_list, wei_pred_scores_list, image_shapes):
            bboxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            
            
            _ , wlabels = torch.max(wscores,dim=1)
            wlabels = wlabels.view(-1, 1)*torch.ones((wscores.shape[0],wscores.shape[1]), device=device)
            
            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            
    
            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            wscores = wscores.reshape(-1)
            labels = labels.reshape(-1)
            wlabels = wlabels.reshape(-1)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, wscores, labels, wlabels = boxes[inds], scores[inds], wscores[inds], labels[inds], wlabels[inds]
            
            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, wscores, labels, wlabels = boxes[keep], scores[keep], wscores[keep], labels[keep], wlabels[keep]
            

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, wscores, labels, wlabels = boxes[keep], scores[keep], wscores[keep], labels[keep], wlabels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_wscores.append(wscores)
            all_labels.append(labels)
            all_wlabels.append(wlabels)

        return all_boxes, all_scores, all_wscores, all_labels, all_wlabels
    
    def postprocess_detections_weirules(self, class_logits, box_regression, proposals, image_shapes, weirules_model=None, deep_inp=None, rules_inp=None):

        device = class_logits.device
        num_classes = class_logits.shape[-1]

        
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        
        
        pred_scores = F.softmax(class_logits, -1)

   
        if weirules_model!=None:
            wei_logits = weirules_model.forward_model(deep_inp, rules_inp)
            #wei_pred_scores = F.normalize(wei_logits, p=1, dim=1)
            wei_pred_scores = F.softmax(wei_logits, dim=1)
#             wei_pred_scores_2 = torch.zeros((wei_pred_scores.shape[0],wei_pred_scores.shape[1]+1))
#             wei_pred_scores_2[:,0]=wei_pred_scores[:,0]
#             wei_pred_scores_2[:,2:]=wei_pred_scores[:,1:]
#             wei_pred_scores = wei_pred_scores_2.cuda()
        # split boxes and scores per image
        if len(boxes_per_image) == 1:
            # TODO : remove this when ONNX support dynamic split sizes
            # and just assign to pred_boxes instead of pred_boxes_list
            pred_boxes_list = [pred_boxes]
            pred_scores_list = [pred_scores]
            wei_pred_scores_list = [wei_pred_scores]
        else:
            pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
            pred_scores_list = pred_scores.split(boxes_per_image, 0)
            wei_pred_scores_list = wei_pred_scores.split(boxes_per_image, 0)


        all_boxes = []
        all_scores = []
        all_labels = []
        all_wscores = []
        all_wlabels = []
        for boxes, scores, wscores, image_shape in zip(pred_boxes_list, pred_scores_list, wei_pred_scores_list, image_shapes):
            bboxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            
            
            _ , wlabels = torch.max(wscores,dim=1)
            wlabels = wlabels.view(-1, 1)*torch.ones((wscores.shape[0],wscores.shape[1]), device=device)
            
            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            
    
            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            wscores = wscores.reshape(-1)
            labels = labels.reshape(-1)
            wlabels = wlabels.reshape(-1)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, wscores, labels, wlabels = boxes[inds], scores[inds], wscores[inds], labels[inds], wlabels[inds]
            
            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, wscores, labels, wlabels = boxes[keep], scores[keep], wscores[keep], labels[keep], wlabels[keep]
            
            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, wscores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, wscores, labels, wlabels = boxes[keep], scores[keep], wscores[keep], labels[keep], wlabels[keep]
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_wscores.append(wscores)
            all_labels.append(labels)
            all_wlabels.append(wlabels+1)

        return all_boxes, all_wscores, all_wlabels
    
    
    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        # type: (Tensor, Tensor, List[Tensor], List[Tuple[int, int]])
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        
        pred_scores = F.softmax(class_logits, -1)

 
        # split boxes and scores per image
        if len(boxes_per_image) == 1:
            # TODO : remove this when ONNX support dynamic split sizes
            # and just assign to pred_boxes instead of pred_boxes_list
            pred_boxes_list = [pred_boxes]
            pred_scores_list = [pred_scores]
        else:
            pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
            pred_scores_list = pred_scores.split(boxes_per_image, 0)


        all_boxes = []
        all_scores = []
        all_labels = []
        all_wscores = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            bboxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            
            
            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            
    
            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            
            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            #keep = keep[:1]            
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels
      
    def forward(self, features, proposals, image_shapes, targets=None, weirules_model=None, rule_input=None, mutual_learning=False, extract_df=False):
        # type: (Dict[str, Tensor], List[Tensor], List[Tuple[int, int]], Optional[List[Dict[str, Tensor]]])
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                

        if self.training:
            proposals, matched_idxs, labels, regression_targets, rule_input_matched = self.select_training_samples(proposals, targets, rule_input)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None
            if rule_input is not None:
                oversampled_rules_inp =[]
                for i in range(len(proposals)):
                    for l in range(proposals[i].shape[0]):
                        oversampled_rules_inp.append(rule_input[i][0])        
                rule_input=torch.stack(oversampled_rules_inp, dim=0)
            else:
                oversampled_rules_inp = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        deep_inp_1=box_features
        box_features = self.box_head(box_features)
        deep_inp_2=box_features
        class_logits, box_regression = self.box_predictor(box_features)


        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            extracted_df = None
            assert labels is not None and regression_targets is not None
            if (rule_input is not None and mutual_learning):
                loss_classifier_f, loss_box_reg, loss_classifier_w = mutual_loss(
                class_logits, box_regression, labels, regression_targets, weirules_model, deep_inp_1, rule_input_matched)
            elif (rule_input is not None and not mutual_learning):
#                 loss_classifier_f, loss_box_reg, loss_classifier_w = weirules_loss_selected(
#                     labels, weirules_model, deep_inp_1, rule_input_matched, class_logits, proposals)  
                loss_classifier_f, loss_box_reg, loss_classifier_w = weirules_loss(
                    labels, weirules_model, deep_inp_1, rule_input_matched)
            else:
                loss_classifier_f, loss_box_reg, loss_classifier_w = fastrcnn_loss(
                    class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier_f": loss_classifier_f,
                "loss_classifier_w": loss_classifier_w,
                "loss_box_reg": loss_box_reg
            }
        else:
            extracted_df = None
            if (extract_df):
                extracted_df = self.postprocess_detections_extract_df(class_logits, box_regression, proposals, image_shapes, deep_inp_1)
            elif (rule_input is None):
                boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
                num_images = len(boxes)
                for i in range(num_images):
                    result.append(
                        {
                            "boxes": boxes[i],
                            "labels": labels[i],
                            "scores": scores[i]
                        }
                    )
            else:
                boxes, scores, labels = self.postprocess_detections_weirules(class_logits, box_regression, proposals, image_shapes, weirules_model, deep_inp_1, rule_input)
                num_images = len(boxes)
                for i in range(num_images):
                    result.append(
                        {
                            "boxes": boxes[i],
                            "labels": labels[i],
                            "scores": scores[i]
                        }
                    )
        return result, losses, extracted_df

'''
pyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''