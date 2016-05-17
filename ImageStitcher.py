import cv2
import numpy as np
import argparse

def stitch(images, ratio=0.75, thresh=4.0):
    
    #Expect images are supplied left to right
    imageL, imageR = images
    
    (keysL, featuresL) = detectFeatures(imageL)
    (keysR, featuresR) = detectFeatures(imageR)
    
    M = matchKeypoints(keysL, keysR, featuresL, featuresR, ratio, thresh)
    if M is None:
        return None
        
    (matches, H, status) = M
    # warp the right image using open cv
    # the function takes the image to be warped, the homography matrix, and the result's shape
    result = cv2.warpPerspective(imageR, H, (imageL.shape[1] + imageR.shape[1], imageR.shape[0]))
    result[0:imageL.shape[0], 0:imageL.shape[1]] = imageL
    
    vis = drawMatches(imageL, imageR, keysL, keysR, matches, status)
    
    return (result, vis)
    
def detectFeatures(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create and call a SIFT featre detector
    detector = cv2.FeatureDetector_create("SIFT")
    keys = detector.detect(gray)
    
    #Create and call a SIFT feature extractor
    extractor = cv2.DescriptorExtractor_create("SIFT")
    (keys, features) = extractor.compute(gray, keys)
    
    #convert keypoits to float Numpy array
    keys = np.float32([kp.pt for kp in keys])
    
    return (keys, features)
    
def matchKeypoints(keysL, keysR, featuresL, featuresR, ratio, thresh):
    
    #Create and run a descriptor matcher
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresR, featuresL, 2)
    matches = []
    
    #prune false positives by testing matches are within a ratio distance to each other
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    
    #compute homography matrix H
    if len(matches) > 4:
        pointsR = np.float32([keysR[i] for (_,i) in matches])
        pointsL = np.float32([keysL[i] for (i,_) in matches])
           
        (H, status) = cv2.findHomography(pointsR, pointsL, cv2.RANSAC, thresh)   
            
        return (matches, H, status)
    
    return None
    
def drawMatches(imageL, imageR, keyL, keyR, matches, status):
    heightL, widthL = imageL.shape[:2]
    heightR, widthR = imageR.shape[:2]
    
    # Set up image to draw on
    vis = np.zeros((max(heightL,heightR), widthL + widthR, 3), dtype="uint8")
    vis[0:heightR, 0:widthR] = imageR
    vis[0:heightL, widthL:] = imageL
    
    # Loop for each pair of points found and draw a line between them
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        if s == 1:
            pointR = (int(keyR[queryIdx][0]), int(keyR[queryIdx][1]))
            pointL = (int(keyL[queryIdx][0]) + widthR, int(keyL[queryIdx][1]))
            cv2.line(vis, pointR, pointL, (0, 0, 255), 1)
    
    return vis

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSCI 4220U Final Project')
    parser.add_argument('LeftFile', help='Left image file')
    parser.add_argument('RightFile', help='Right iamge file')
    args = parser.parse_args()
    
    imageA = cv2.imread(args.LeftFile)
    imageA = cv2.resize(imageA, (640, 480))
    imageB = cv2.imread(args.RightFile)
    imageB = cv2.resize(imageB, (640, 480))
    
    result, vis = stitch([imageA, imageB])
    
    cv2.imshow("Image A", imageA)
    cv2.imshow("Image B", imageB)
    cv2.imshow("Result", result)
    cv2.imshow("Point Matches", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    