import cv2
import numpy as np
from scipy.signal import argrelextrema, find_peaks
import math

class Side:
    def __init__(self, points):
        self.points = points
        x1, y1 = (points[0][0], points[0][1])
        x2, y2 = (points[-1][0], points[-1][1])
        dx = x1-x2
        dy = y1-y2
        self.length = math.sqrt(dx*dx+dy*dy)
        #print("x1 %d, x2 %d , y1 %d y2 %d length: %d" % ( x1, x2, y1, y2, self.length) )
        self.normP = []
        self.norm100 = []
        for p in points:
            x0, y0 = (p[0], p[1])
            d = ((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / self.length
            self.normP.append(d)
        for i in range(100):
            ind = min( len(self.normP)-1, round(i/100*len(self.normP)))
            self.norm100.append(self.normP[ind])

    def match(self, s):
        diff = 0
        if abs(self.length-s.length) > 10:
            return 10 
        for i in range(100):
            j = 99-i
            diff += pow(self.norm100[i]+s.norm100[j], 2)
        diff /= 100
        if diff < 6:
            print("diff: %.1f %d %d" % (diff, self.length, s.length) )
            for i in range(100):
                j = 99-i
                print( "%.2f %.2f %.2f" % (self.norm100[i], s.norm100[i], -s.norm100[j]) )
        return diff

class Puzzle:
    def __init__(self):
        self.sides = []
    def match(self, p):
        bestDiff = 1e5
        bestI, bestJ  = 0, 0
        for i in range(len(p.sides)):
            for j in range(len(self.sides)):
                diff = p.sides[i].match(p.sides[j])
                if diff < bestDiff:
                    bestDiff = diff
                    bestI = i
                    bestJ = j
        print("best diff %.1f" % bestDiff)
        return ( bestDiff, bestI, bestJ )


# Функция для загрузки и обработки изображения
def load_and_preprocess(image_path):
    img = cv2.imread(image_path)
    print("ndim ", img.ndim, img.shape )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    bin = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,301,-50)
    print("bin ", bin.ndim, bin.shape )
    #ADAPTIVE_THRESH_GAUSSIAN_C
    return img, bin

# Функция для поиска контуров деталей пазла
def find_puzzle_pieces2(edges):
    print("len edges:", edges.shape)
    contours,hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]

    summ_len, count = (0, 0)
    for i in contours:
        l = len(i)
        if l<100:
            continue

        summ_len += l
        count += 1
        #print(len(i))
    if count == 0:
        return contours

    av_size = summ_len/count
    filtered_cont = []
    for i in contours:
        l = len(i)
        if l<100 or l<av_size*0.6 or l> av_size*1.3:
            continue
        filtered_cont.append(i)
    img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img, filtered_cont, -1, (0, 255, 0), 3)

    resultPuzzles = []
    d1, d2 = ( 5, 10 )
    for c in filtered_cont:
        size = len(c)
        flags_arr = []
        avX, avY = ( 0, 0 )
        for p in c:
            avX += p[0][1]
            avY += p[0][0]
        avX /= size
        avY /= size

        dist_array = []
        for p in c:
            p = p[0]
            dist = math.sqrt(math.pow(p[1]-avX, 2) + math.pow(p[0]-avY,2))
            dist_array.append(round(dist))
        print()
        globalAverage = np.average(dist_array)
        minInd = np.argmin(dist_array)
        da = [0] * len(dist_array)
        print(len(da), len(dist_array))#1
        da[0:]=dist_array[minInd:]
        da.extend(dist_array[:minInd])
        #da - смещен то есть чтобы получить индекс minInd в оригинале соответсвует 0, то есть к da надо прибавить minInd
        maxInd = np.argmax(da)

        (peaksMax, propertiesMax) = find_peaks(da, prominence=6, distance=10)#, threshold=6)#

        print("peaks:", peaksMax)

        delta_arr = []
        for k in range(len(peaksMax)-1):
            if len(peaksMax) == 8 and k == maxInd:
                printf("skip")
                continue
            delta_arr.append([peaksMax[k],(peaksMax[k+1]-peaksMax[k])/size*100])
            print("%d %.2f" % ( peaksMax[k], delta_arr[-1][1]) )
        delta_arr.append([peaksMax[-1],(size-peaksMax[-1]+peaksMax[0])/size*100])
        print("%d %.2f" % ( peaksMax[-1], (delta_arr[-1][1]) ))
        #print(delta_arr[0])

        bestInd = 0
        bestVal = 100
        for k in range(len(delta_arr)):
            val = delta_arr[k][1]
            d = abs(25-val)
            if d < bestVal:
                bestVal = d
                bestInd = k

        s = len(delta_arr)
        print("best index %d best val %.2f  len: %d   minIndex %d" % (bestInd, bestVal, s, minInd))
        index_arr = []
        # теперь надо положить 1ый индекс, который для лучшей дельты, и потом класть те которые в интервале от 20 до 30

        curr_delta = delta_arr[bestInd]
        for k in range(1,s):
            i = (k+bestInd) % s
            #if curr_delta >= 18 and curr_delta <= 33:
            if curr_delta[1] >= 20 and curr_delta[1] <= 30:
                index_arr.append(curr_delta)
                curr_delta = delta_arr[i]
            else:
                curr_delta[1] += delta_arr[i][1]
        index_arr.append(curr_delta)

        if len(index_arr) != 4:
            continue
        print(index_arr)
        puzzle = Puzzle()
        s = len(dist_array)
        for k in range(4):
            points = []
            indStart = index_arr[k][0]
            indEnd = index_arr[(k+1)%4][0]
            print("original %d   da %d" % ((indStart+minInd)%s, indStart) )
            if indEnd < indStart:
                indEnd += s
            for j in range(indStart,indEnd):
                ind = (j+minInd)%s
                p = [c[ind][0][0],c[ind][0][1]]
                points.append(p)
            side = Side(points)
            puzzle.sides.append(side)
        resultPuzzles.append(puzzle)
        # 1 найти наиболее близкий к 25 в диапазона 25-30, если нет такого, то пропускае этот элемент 
        # по хорошему, надо найти вариант, который начинает с 45 градусов, либо просто игнорируем самую большую вершину, 
        # и дальше по основному алгоритмуа остальные сч

        print()
    cv2.imshow('img',img)
    return resultPuzzles

def sortFunc(e):
  return e["diff"]

def matchPuzzles(puzzles):
    result = []
    for i in range(len(puzzles)):
        bestDiff = 1e5
        bestPuzzleIndex, bestSideI, bestSideJ  = 0, 0, 0
        for j in range(i+1,len(puzzles)):
            (diff, bi, bj) = puzzles[i].match(puzzles[j])
            if (diff < bestDiff):
                bestDiff = diff
                bestPuzzleIndex, bestSideI, bestSideJ = j, bi, bj
                #print(i, j, diff)
                #result.append({"diff" : diff, "i" : i, "j" : j, "bi" : bi, "bj" : bj })
        result.append({"diff" : bestDiff, "i" : i, "j" : bestPuzzleIndex, "bi" : bestSideI, "bj" : bestSideJ })
    result.sort(key=sortFunc)
    return result

def drawContour(img, cont, color, radius):
    for p in cont:
        cv2.circle(img, p, radius, color, -1)

def draw(bin, puzzles, result):
    for it in result:
        cont1 = puzzles[it["i"]].sides[it["bi"]].points
        cont2 = puzzles[it["j"]].sides[it["bj"]].points
        img = cv2.cvtColor(bin, cv2.COLOR_GRAY2BGR)
        drawContour(img, cont1, (0, 255, 0), 3)
        drawContour(img, cont2, (0, 0, 255), 3)
        print(it["diff"])
        cv2.imshow("Puzzle Connections", img)
        ch = cv2.waitKey(0)
        if ch == 27:
            break

# Основная функция
def main(image_path):
    img, bin = load_and_preprocess(image_path)

    puzzles = find_puzzle_pieces2(bin)
    result = matchPuzzles(puzzles)
    draw(bin, puzzles, result)

    # Отображение результата
    cv2.imshow("Puzzle Connections", colorBin)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()

# Запуск кода с изображением
image_path = "00.png"  # Укажите путь к изображению
cv2.namedWindow('img', 0)
cv2.namedWindow('Puzzle Connections', 0)
main(image_path)