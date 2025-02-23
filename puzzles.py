import cv2
import numpy as np
from scipy.signal import argrelextrema, find_peaks
import math


class Side:
    def __init__(self, points):
        #print(points)
        self.points = points
        x1, y1 = (points[0][0], points[0][1])
        x2, y2 = (points[-1][0], points[-1][1])
        dx = x1-x2
        dy = y1-y2
        self.length = math.sqrt(dx*dx+dy*dy)
        print("x1 %d, x2 %d , y1 %d y2 %d length: %d" % ( x1, x2, y1, y2, self.length) )
        self.normP = []
        self.norm100 = []
        for p in points:
            x0, y0 = (p[0], p[1])
            #print(x0, y0)
            #d = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1 ) / self.length
            d = ((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / self.length
            self.normP.append(d)
        for i in range(100):
            ind = min( len(self.normP)-1, round(i/100*len(self.normP)))
            #print( "%.2f " % self.normP[ind], end="" )
            self.norm100.append(self.normP[ind])
        #print()

    def match(self, s):
        diff = 0
        if abs(self.length-s.length) > 10:
            return 10 
        for i in range(100):
            j = 99-i
            #diff += abs(self.norm100[i]+s.norm100[j])
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
    #edges = cv2.Canny(blurred, 50, 150)

    bin = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,301,-50)
    print("bin ", bin.ndim, bin.shape )
    #ADAPTIVE_THRESH_GAUSSIAN_C
    return img, bin#, edges

# Функция для поиска контуров деталей пазла
'''
def find_puzzle_pieces(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def correlationWith(src1, template):
    av_templ = 0
    s = len(template)

    for st in range(src1):
        for i in range(template):

def getNormalizeArray(template):
    avTH = np.average(template)
    stdTH = np.std(template)
    newT = (template-avTH)/stdTH
    print(T)
    return T

def isSameAngles( angle1, angle2, delta ):
    d = abs( angle1 - angle2 )
    if d <= delta or \
        abs(delta-360) <= delta or \
        abs(delta+360) <= delta:
        return True
        #math.abs(delta-180) <= delta or
        #math.abs(delta+180) <= delta or
    return False

def normalizeAngle(delta):
    if delta < -180:
        delta += 180
    if delta > 180:
        delta -= 180
    return delta
'''

# Функция для поиска контуров деталей пазла
def find_puzzle_pieces2(edges):
    print("len edges:", edges.shape)
    contours,hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print("contour: ", contours[1])

    cnt = contours[0]
    #hull = cv2.convexHull(cnt,returnPoints = False)
    #defects = cv2.convexityDefects(cnt,hull)
    #print(defects)

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
            #print( p[0][0], p[0][1])
        avX /= size
        avY /= size

        dist_array = []
        #angle_array = []
        for p in c:
            p = p[0]
            #angle = math.degrees( math.atan2(p[1]-avX, p[0]-avY) )+180
            dist = math.sqrt(math.pow(p[1]-avX, 2) + math.pow(p[0]-avY,2) )
            #angle_stat[round(angle)] += 1
            dist_array.append(round(dist))
            #print( len(dist_array)-1, round(dist))
            #print("(%d, %d) " % (p[0], p[1]), end="" )
            #angle_array.append( round(angle) )
        print()
        #print(dist_array)
        #print(angle_array)
        #print(angle_stat)
        #topCorrel = np.correlate(dist_array, topHill, "full")
        #for i in topCorrel:
        #    print("%.2f " % i, end='')
        #print()
        #for p in range(size):
        #print(topCorrel)
        globalAverage = np.average(dist_array)
        minInd = np.argmin(dist_array)
        da = [0] * len(dist_array)
        print(len(da), len(dist_array))#1
        da[0:]=dist_array[minInd:]
        da.extend(dist_array[:minInd])

        #for p in range(len(da)):
        #    print( p, da[p])
        #da - смещен то есть чтобы получить индекс minInd в оригинале соответсвует 0, то есть к da надо прибавить minInd
        #da[0]=dist_array[minInd]
        #print(da)
        #print(len(da), len(dist_array))
        maxInd = np.argmax(da)

        (peaksMax, propertiesMax) = find_peaks(da, prominence=6, distance=10)#, threshold=6)#
        #(peaksMin, propertiesMin) = find_peaks(dist_array, prominence=6, distance=10)#, threshold=6)#

        print("peaks:", peaksMax)
        #print(propertiesMax)

        delta_arr = []
        for k in range(len(peaksMax)-1):
            if len(peaksMax) == 8 and k == maxInd:
                printf("skip")
                continue
            #if k == len(peaksMax):
            #    proc = (peaksMax[k+1]-peaksMax[k])/size*100
            #else:
            #    proc = (peaksMax[k+1]-peaksMax[k])/size*100
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
        index_arr2 = []
        #curr_proc = 0
        # еперь надо положить 1ый индекс, который для лучшей дельты, и потом класть те которые в интервале от 20 до 30
        # или 17 до 33
        #index_arr.append(delta_arr[bestInd])
        #prevBestInd = (bestInd-1+s) % s
        #index_arr2.append(peaksMax[prevBestInd])
        curr_delta = delta_arr[bestInd]

        #iPrev = bestInd
        for k in range(1,s):
            i = (k+bestInd) % s
            #if curr_delta >= 18 and curr_delta <= 33:
            if curr_delta[1] >= 20 and curr_delta[1] <= 30:
                #index_arr.append(curr_delta)
                index_arr.append(curr_delta)
                #index_arr2.append(peaksMax[iPrev])
                curr_delta = delta_arr[i]
                #iPrev = (k+bestInd) % s
            else:
                curr_delta[1] += delta_arr[i][1]
        index_arr.append(curr_delta)

        if len(index_arr) != 4:
            continue
        print(index_arr)#, index_arr2)
        puzzle = Puzzle()
        s = len(dist_array)
        for k in range(4):
            points = []
            indStart = index_arr[k][0]
            indEnd = index_arr[(k+1)%4][0]
            print("original %d   da %d" % ((indStart+minInd)%s, indStart) )
            if indEnd < indStart:
                indEnd += s
            #print("indStart %d indEnd %d" % (indStart,indEnd))
            for j in range(indStart,indEnd):
                ind = (j+minInd)%s
                #p = c[ind][0]
                p = [c[ind][0][0],c[ind][0][1]]
                points.append(p)
            side = Side(points)
            puzzle.sides.append(side)
            #side.create()
            #resPoints.append(points)
        resultPuzzles.append(puzzle)
        # 1 найти наиболее близкий к 25 в диапазона 25-30, если нет такого, то пропускае этот элемент 
        # по хорошему, надо найти вариант, который начинает с 45 градусов, либо просто игнорируем самую большую вершину, 
        # и дальше по основному алгоритмуа остальные сч
        #for it in flags_arr:
        #    print( it, end=" " )

        print()

    '''
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        cv.line(img,start,end,[0,255,0],2)
        cv.circle(img,far,5,[0,0,255],-1)
    '''
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
        #cv2.drawContours(img, [cont1], -1, (0, 255, 0), 3)
        #cv2.drawContours(img, [cont2], -1, (0, 0, 255), 3)
        print(it["diff"])
        cv2.imshow("Puzzle Connections", img)
        ch = cv2.waitKey(0)
        if ch == 27:
            break
'''
# Функция для определения возможных соединений между деталями
def match_puzzle_pieces(contours):
    connections = []
    for i, cnt1 in enumerate(contours):
        for j, cnt2 in enumerate(contours):
            if i != j:
                dist = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I1, 0.0)
                if dist < 0.1:  # Порог схожести
                    connections.append((i, j))
    return connections

# Функция для визуализации соединения
def visualize_connections(img, contours, connections):
    for i, j in connections:
        c1 = np.mean(contours[i], axis=0)[0]
        c2 = np.mean(contours[j], axis=0)[0]
        cv2.line(img, tuple(c1.astype(int)), tuple(c2.astype(int)), (0, 255, 0), 2)
    return img
'''
# Основная функция
def main(image_path):
    img, bin = load_and_preprocess(image_path)
    #edges = load_and_preprocess(image_path)

    puzzles = find_puzzle_pieces2(bin)
    result = matchPuzzles(puzzles)
    draw(bin, puzzles, result)
    #contours = find_puzzle_pieces(edges)
    #connections = match_puzzle_pieces(contours)
    #output_img = visualize_connections(img, contours, connections)

    # Отображение результата
    cv2.imshow("Puzzle Connections", colorBin)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()

# Запуск кода с изображением
image_path = "00.png"  # Укажите путь к изображению
cv2.namedWindow('img', 0)
cv2.namedWindow('Puzzle Connections', 0)
main(image_path)