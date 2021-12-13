class Finger:
    def __init__(self, lmList,coor):
        # position 1 at the very end pos 4 at the very base
        self.coord_1 = lmList[8][1:]  # x,y

        self.coord_2 = lmList
        self.coord_3 = lmList
        self.coord_4 = lmList

        # self.coord_wrist = pos_wrist
        # self.img = img

    def expose(self):
        print(self.coord_1)

    def CheckForNote(self):
        pass

    def drawCircles(self):
        cv2.circle(self.img, (self.coord_1[0], self.coord_1[1]), 4, (255, 0, 0), cv2.FILLED)



index.expose()

class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

  def myfunc(self):
    print("Hello my name is " + self.name)

p1 = Person("John", 36)
p1.myfunc()
