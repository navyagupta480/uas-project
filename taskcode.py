import cv2
import numpy as np
import os

# capacities for pads
def getPadCapacity(pad):
    if pad == "pink":
        return 3
    elif pad == "blue":
        return 4
    else:
        return 2

# detect land and water
def cutLandWater(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ocean = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
    land = cv2.inRange(hsv, (20, 40, 40), (85, 255, 255))
    out = img.copy()
    out[ocean > 0] = (255, 0, 0)
    out[land > 0] = (0, 255, 0)
    return out, ocean, land

# detect people (casualties)
def findPeople(img):
    people = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            continue
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        # shapes
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            shape = "square"
        else:
            shape = "star"

        # colors
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        h, s, v, _ = cv2.mean(hsv, mask=mask)

        if (h < 15 or h > 165) and s > 50:
            color = "red"
        elif 15 <= h < 40:
            color = "yellow"
        elif 40 <= h < 90:
            color = "green"
        else:
            color = "unknown"

        people.append({"shape": shape, "color": color, "coords": (cx, cy)})

    return people

# detect pads
def findPads(img):
    pads = {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=50, param2=20, minRadius=10, maxRadius=200
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i, (x, y, r) in enumerate(circles[0, :]):
            mask = np.zeros(gray.shape, np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            h, s, v, _ = cv2.mean(hsv, mask=mask)
            if (h < 15 or h > 165) and s > 50:
                color = "pink"
            elif 90 < h < 130:
                color = "blue"
            elif v < 80:
                color = "grey"
            else:
                color = f"pad{i}"
            pads.setdefault(color, []).append((x, y))
    return pads

# assign people to pads
def divideWork(people, pads):
    assignments = {p: [] for p in pads.keys()}
    for person in people:
        best = None
        bestScore = -1
        for pad, coords in pads.items():
            dist = np.linalg.norm(np.array(person["coords"]) - np.array(coords[0]))
            score = 1 / (dist + 1e-6)
            if len(assignments[pad]) < getPadCapacity(pad):
                if score > bestScore:
                    bestScore = score
                    best = pad
        if best:
            assignments[best].append(person)
    return assignments

# main process
def processImage(path, savePath):
    img = cv2.imread(path)
    if img is None:
        print("Image not found:", path)
        return
    segmented, _, _ = cutLandWater(img)
    people = findPeople(img)
    pads = findPads(img)
    assignments = divideWork(people, pads)

    out = img.copy()
    for p in people:
        cv2.circle(out, p["coords"], 10, (0, 0, 255), -1)
    for pad, coords in pads.items():
        for c in coords:
            cv2.circle(out, c, 15, (255, 255, 0), 2)
    for pad, plist in assignments.items():
        for p in plist:
            cv2.line(out, p["coords"], pads[pad][0], (0, 255, 0), 2)

    cv2.imwrite(savePath, out)
    print("Saved processed image to", savePath)

# run for all images in folder
if __name__ == "__main__":
    folder = "/Users/navyagupta/taskimages"  
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(folder):
        if file.endswith((".png", ".jpg")):
            path = os.path.join(folder, file)
            save_path = os.path.join(output_folder, f"out_{file}")
            processImage(path, save_path)


