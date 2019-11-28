# How to label

## Images
The input images have two parts:
- the left-most image is the transformed image. The black areas in the transformed image are not in the original image. It is turned 45 degrees counterclockwise. 
- the right-most image is the untransformed image. All the irrelevant areas are taken out by the transformation. Unfortunately, parts of the stage are not in the camera. So these areas cannot be present in the transformed image.

## Labeling
The images are 64 by 64 pixels. You should label the x- and y-coordinates of the center of the red and blue agents in the transformed images. Please label from (1, 1) at the top-left corner up to and including (64, 64) at the bottom-right corner.

## Hidden agents
If an agent is completely in a black area (outside the shot of the camera), then you cannot properly label the coordinates of that agent. The blue agent follows the red agent around. So you should be able to discern in which of the three black areas the red or blue agent is hiding, by how close the blue or red agent is to one of the corners. Please label such images with the coordinates of the corner ((1, 1), (64, 1), or (1, 64)).

## Overlapping agents
Lastly, it would be nice if you noted whenever the two agents are overlapping. This information might be used to train the model more effectively.

## Sheets
The Google Sheets file can be found [here](https://docs.google.com/spreadsheets/d/1evDrBCiAOz2T-c8YuZ_rBR-cBUegqDVKjAEKWHyJjKE/edit?usp=sharing).

## Division of Labour
I will do image 0 to 332, Maurice will do image 333 to 665 and Roel will do image 666 to 999.
