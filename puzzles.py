import cv2
import numpy as np
from scipy.signal import argrelextrema, find_peaks
import math

MAX_DIFF=1e5

class Side:
    def __init__(self, points):
        self.points = points
        x1, y1 = (points[0][0], points[0][1])
        x2, y2 = (points[-1][0], points[-1][1])
        dx = x1-x2
        dy = y1-y2
        self.length = math.sqrt(dx*dx+dy*dy)
        self.normP = []
        self.norm100 = []
        for p in points:
            x0, y0 = (p[0], p[1])
            d = ((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)# / self.length
            self.normP.append(d)
        for i in range(100):
            ind = min( len(self.normP)-1, round(i/100*len(self.normP)))
            self.norm100.append(self.normP[ind])

    def match(self, s):
        diff = 0
        if abs(self.length-s.length) > 5:
            return MAX_DIFF
        for i in range(100):
            j = 99-i
            diff += abs(self.norm100[i]+s.norm100[j])
        diff /= 100
        return diff

class Puzzle:
    def __init__(self):
        self.sides = []
        #self.sideShift = 0 
    def match(self, p):
        bestDiff = MAX_DIFF
        bestI, bestJ  = 0, 0
        for i in range(len(self.sides)):
            for j in range(len(p.sides)):
                diff = self.sides[i].match(p.sides[j])
                if diff < bestDiff:
                    bestDiff = diff
                    bestI = i
                    bestJ = j
        #print("best diff %.1f" % bestDiff)
        return ( bestDiff, bestI, bestJ )

    def match2(self, puzzles, indSide, indStart):
        bestDiff = MAX_DIFF
        bestJ, bestSideJ  = 0, 0
        for j in range(indStart,len(puzzles)):
            p = puzzles[j]
            for jj in range(len(p.sides)):
                diff = self.sides[indSide].match(p.sides[jj])
                if diff < bestDiff:
                    bestDiff = diff
                    bestJ = j
                    bestSideJ = jj
        #print("best diff %.1f" % bestDiff)
        return ( bestDiff, bestJ, bestSideJ )

    def sort(self):
        glX, glY = 0, 0
        for i in range(4):
            glX += self.sides[i].points[0][0]
            glY += self.sides[i].points[0][1]

        temp = []
        glX /= 4
        glY /= 4
        for it in self.sides:
            avX, avY = 0, 0
            for p in it.points:
                avX += p[0]
                avY += p[1]
            avX /= len(it.points)
            avY /= len(it.points)
            angle = math.atan2(avY-glY, avX-glX)
            if angle < -math.pi*3/4 or angle > math.pi*3/4:
                angle = -math.pi
            it.angle = angle
            temp.append([angle, it])

        temp.sort(key=lambda x:x[0])
        self.sides = []
        for it in temp:
            self.sides.append(it[1])

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
        #print()
        globalAverage = np.average(dist_array)
        minInd = np.argmin(dist_array)
        da = [0] * len(dist_array)
        #print(len(da), len(dist_array))#1
        da[0:]=dist_array[minInd:]
        da.extend(dist_array[:minInd])
        #da - смещен то есть чтобы получить индекс minInd в оригинале соответсвует 0, то есть к da надо прибавить minInd
        maxInd = np.argmax(da)

        (peaksMax, propertiesMax) = find_peaks(da, prominence=6, distance=10)#, threshold=6)#

        delta_arr = []
        for k in range(len(peaksMax)-1):
            if len(peaksMax) == 8 and k == maxInd:
                print("skip")
                continue
            delta_arr.append([peaksMax[k],(peaksMax[k+1]-peaksMax[k])/size*100])
            #print("%d %.2f" % ( peaksMax[k], delta_arr[-1][1]) )
        delta_arr.append([peaksMax[-1],(size-peaksMax[-1]+peaksMax[0])/size*100])
        #print("%d %.2f" % ( peaksMax[-1], (delta_arr[-1][1]) ))
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
        #print("best index %d best val %.2f  len: %d   minIndex %d" % (bestInd, bestVal, s, minInd))
        index_arr = []
        # теперь надо положить 1ый индекс, который для лучшей дельты, и потом класть те которые в интервале от 20 до 30

        curr_delta = delta_arr[bestInd]
        for k in range(1,s):
            i = (k+bestInd) % s
            if curr_delta[1] >= 20 and curr_delta[1] <= 30:
                index_arr.append(curr_delta)
                curr_delta = delta_arr[i]
            else:
                curr_delta[1] += delta_arr[i][1]
        index_arr.append(curr_delta)

        if len(index_arr) != 4:
            continue
        #print(index_arr)
        puzzle = Puzzle()
        s = len(dist_array)
        for k in range(4):
            points = []
            indStart = index_arr[k][0]
            indEnd = index_arr[(k+1)%4][0]
            if indEnd < indStart:
                indEnd += s
            for j in range(indStart,indEnd):
                ind = (j+minInd)%s
                p = [c[ind][0][0],c[ind][0][1]]
                points.append(p)
            side = Side(points)
            puzzle.sides.append(side)

        puzzle.sort()
        resultPuzzles.append(puzzle)
        # 1 найти наиболее близкий к 25 в диапазона 25-30, если нет такого, то пропускае этот элемент 
        # по хорошему, надо найти вариант, который начинает с 45 градусов, либо просто игнорируем самую большую вершину, 
        # и дальше по основному алгоритмуа остальные сч

        #print()
    cv2.imshow('img',img)
    return resultPuzzles

def matchPuzzles(puzzles):
    result = []
    for i in range(len(puzzles)):
        print("process match %.1f %%" % (100*(i+1)/len(puzzles)), end='\r')
        p = puzzles[i]
        for ii in range(4):
            (diff, bestJ, bestSideJ) = puzzles[i].match2(puzzles, ii, i+1)
            if diff == MAX_DIFF:
                continue
            # это matchResult
            result.append({"diff" : diff,
                "i" : i, "j" : bestJ,# index puzzle
                "si" : ii, "sj" : bestSideJ,# si side for i, sj - side for j
                "imI" : puzzles[i].imageIndex, "imJ" : puzzles[bestJ].imageIndex })
    print()
    result.sort(key=lambda x:x["diff"])
    return result

def getNeighbours(indIm, indPz, matchResult):
    neighbours = []
    for it in matchResult:
        if indIm == it["imI"] and indPz == it["i"]:
            neighbours.append([it["si"], it["imJ"], it["j"], it["sj"], it["diff"]])
        if indIm == it["imJ"] and indPz == it["j"]:
            neighbours.append([it["sj"], it["imI"], it["i"], it["si"], it["diff"]])
        # 0=side index, 1=image index, 2=puzzle index, 3=side index, 4=diff
        # возвращаем всех соседей, которых нашли для этого пазла
    return neighbours

def getDeltaPos(ind):
    return {
        0: (-1, 0),
        1: (0, -1),
        2: (1, 0),
        3: (0, 1),
    }.get(ind)

# pos - позиция, где находится новый элемент относительно старого
# ind - какой стороной новый элемент граничит
def getSideShift(posOld, ind):
    #shift = 6-posOld
    #return (shift-ind)%4
    shift = 6 if posOld == 0 else posOld+2
    return (shift-ind)%4

def checkIsAlreadyProcessed(val, alreadyProcess):
    for it in alreadyProcess:
        if val[0] == it[0] and val[1] == it[1] and val[2] == it[2]:
            return True
    return False

class Field:
    def __init__(self):
        #self.arr = np.matrix(size=(100,100), dtype=int)
        self.arr = np.zeros([100,100], dtype=int)
        self.shiftX = 50
        self.shiftY = 50
        self.lastIndex = 1

    def add(self, x, y):
        #print("--------- x=%d y=%d   (%d, %d )" % (x,y, x+self.shiftX, y+self.shiftY))
        self.arr[y+self.shiftY][x+self.shiftX]=self.lastIndex
        self.lastIndex += 1

    def print(self):
        #print(self.arr.shape[0], self.arr.shape[1])
        #print("lastIndex: ", self.lastIndex)
        minX, maxX, minY, maxY = (self.shiftX*2, 0, self.shiftX*2, 0)
        for j in range(self.arr.shape[0]):
            for i in range(self.arr.shape[1]):
                if self.arr[j][i] != 0:
                    minX = min(minX, i)
                    maxX = max(maxX, i)
                    minY = min(minY, j)
                    maxY = max(maxY, j)
        #print("minX %d, maxX %d, minY %d, maxY %d" % (minX, maxX, minY, maxY))
        for j in range(minY, maxY+1):
            for i in range(minX, maxX+1):
                #print(self.arr[j][i], end ='', sep='')
                if self.arr[j][i] == 0:
                    print(" ", end ='', sep='')
                #else:
                #    print(self.arr[j][i], end ='', sep='')
                elif self.arr[j][i] == self.lastIndex-1:
                    print("o", end ='', sep='')
                else:
                    print("X", end ='', sep='')
            print()

def makeAllPicture(images, puzzles, matchResult):
    alreadyProcess = []
    field = [[0, 0]]
    f = Field()
    f.add(0, 0)
    fe = matchResult[0]
    currentArray = [ [ 0, fe["imI"], fe["i"], 0, 0, 0, 0 ]]#pos (0, 0)
    #currentArray = [ matchResult[0] ]
    while len(currentArray) > 0:
        newArray = []
        print( "process array size:", len(currentArray))
        # currentArray - текущий массив пазлов для которых мы ищем соседей
        for it in currentArray:
            glX, glY = it[4], it[5]
            srcShift = it[6]
            # нашли соседей для одного пазла - надо проверить
            neighbours = getNeighbours(it[1], it[2], matchResult)
            # 1) у нас может быть несколько вариантов на одну сторону - надо проигнорировать те, где уже установили
            # 2) там уже может что то cтоять
            processSides = []
            for j in neighbours:
                # neighbours: it["sj"], it["imI"], it["i"], it["si"], it["diff"]
                # 0 для какой стороны пред/ 1 инд изображение / 2 инд пазла / 3 номер стороны текщий/ 4 разница
                # TODO: 3 - и вот сюда уже надо плюсовать поворот исходного пазла
                # it[0] - исходная сторона которая предыдущего пазла была использована ранее, уже с учетом поворота(отрисовка)
                # it[3] - это предыдущее j[3]
                # it[6] - это предыдущий sideShift = getSideShift(sideInd, it[3])
                # TODO: а должна быть сумма всех предыдущих поворотов
                # j[0] - source side index
                # j[3] - destination side index
                # 0=side index, 1=image index, 2=puzzle index, 3=side index, 4=diff
                sideInd = (j[0]+srcShift)%4
                # + нужна проверка что с этой стороны уже не стоит пазл
                if checkIsAlreadyProcessed([sideInd, j[1], j[2]], alreadyProcess) or (sideInd in processSides):
                    continue
                dx, dy = getDeltaPos(sideInd)
                # sideShift - насколько новый элемент надо будет повернуть
                sideShift = getSideShift(sideInd, j[3])#it[3])
                x, y = glX+dx, glY+dy
                #print("old %d new %d rotate %d" % (sideInd, j[3], sideShift))
                print("imI=%d imJ=%d   glX=%d glY=%d dx=%d dy=%d    src shift %d  old/new si %d/%d  = old/new sj" % 
                      (it[1],j[1], glX,glY, dx,dy, sideInd, j[3], sideShift))#srcShift, j[0], sideInd ))
                if [x,y] in field:
                    continue

                print("neighbour:", j)
                dataForDraw = {"diff" : j[4],
                    "i" : it[2], "j" : j[2],# index puzzle
                    "si" : j[0], "sj" : j[3],
                    "imI" : it[1], "imJ" : j[1] }

                alreadyProcess.append( [sideInd, j[1], j[2]] )#
                field.append([x, y])
                f.add(x, y)
                processSides.append(sideInd)
                #newArray.append(j)
                f.print()
                if not draw(images, puzzles, dataForDraw):
                    return
                
                # j[1] - индекс изображения
                # j[2] - индекс пазла
                # j[3] - индекс стороны
                # TODO: j[3] заменить на sideShift?
                newArray.append([sideInd, j[1], j[2], j[3], x, y, sideShift])

            #neighbours.append([it["si"], it["imJ"], it["j"], it["sj"], it["diff"]])
            #neighbours.append([it["sj"], it["imI"], it["i"], it["si"], it["diff"]])
            #print(neighbours)
        f.print()
        #print(field)
        currentArray = newArray

def drawContour(img, cont, color, radius):
    for p in cont:
        cv2.circle(img, p, radius, color, -1)

def draw(images, puzzles, it):
    #for it in result:
        cont1 = puzzles[it["i"]].sides[it["si"]].points
        cont2 = puzzles[it["j"]].sides[it["sj"]].points

        bin1 = images[it["imI"]]
        img1 = cv2.cvtColor(bin1, cv2.COLOR_GRAY2BGR)
        if it["imI"] != it["imJ"]:
            bin2 = images[it["imJ"]]
            img2 = cv2.cvtColor(bin2, cv2.COLOR_GRAY2BGR)
        else:
            img2 = img1

        drawContour(img1, cont1, (0, 255, 0), 5)
        drawContour(img2, cont2, (0, 0, 255), 5)
        print(it["diff"])
        cv2.imshow("src", img1)
        if it["imI"] != it["imJ"]:
            cv2.imshow("new", img2)
        ch = cv2.waitKey(0)
        if ch == 27:
            return False
        return True

def main(paths):
    puzzles = []
    images = []
    for i in range(len(paths)):
        imagePath = paths[i]
        img, bin = load_and_preprocess(imagePath)
        puzzlesLocal = find_puzzle_pieces2(bin)
        print(imagePath, len(puzzlesLocal))
        for it in puzzlesLocal:
            it.imageIndex = i
        puzzles.extend(puzzlesLocal)
        images.append(bin)

    result = matchPuzzles(puzzles)
    print( "Count puzzles:", len(puzzles), "count matched:", len(result))
    makeAllPicture(images, puzzles, result)
    #for it in result:
    #    draw(images, puzzles, it)

    cv2.destroyAllWindows()

images = ["03.png", "04.png", "05.png"]
cv2.namedWindow('img', 0)
cv2.namedWindow('src', 0)
cv2.namedWindow('new', 0)
main(images)