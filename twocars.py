from ltlcar import *
from PIL import Image

def createImg(c1, c2, i):
    img = Image.new('RGB', (w.dim, w.dim), "white")  # create a new black image
    pixels = img.load()  # create the pixel map
    pixels[c1] = (255, 0, 0)
    pixels[c2] = (0, 0, 255)
    img = img.resize((500, 500))
    img.show(str(i))

if __name__ == '__main__':

    # Define actions
    actions = list()
    direction = [1, 0, -1, 0]
    cos = lambda x: direction[x]
    sin = lambda x: direction[(3 + x) % 4]
    cw = lambda x: (x - 1) % 4
    ccw = lambda x: (x + 1) % 4

    def wait(cell):
        return cell                                                              # wait

    def forward(cell):
        return cell[0] + cos(cell[2]), cell[1] + sin(cell[2]), cell[2]                                # forward

    def fwdRight(cell):
        return cell[0] + sin(cell[2]) + cos(cell[2]), cell[1] + sin(cell[2]) - cos(cell[2]), cell[2] # fr

    def fwdLeft(cell):
        return cell[0] - sin(cell[2]) + cos(cell[2]), cell[1] + sin(cell[2]) + cos(cell[2]), cell[2]  # fl

    def right(cell):
        return cell[0] + sin(cell[2]), cell[1] - cos(cell[2]), cw(cell[2])  # RIGHT

    def left(cell):
        return cell[0] - sin(cell[2]), cell[1] + cos(cell[2]), ccw(cell[2])  # LEFT

    actions = [wait, forward, fwdRight, fwdLeft, right, left]

    # Create world
    w = World(roadMap='world55/road.bmp', dim=5, grassMap='world55/grass.bmp', stopMap='world55/stopsign.bmp')
    print(np.rot90(w.labelMap))

    # Cars
    c1_start = 21
    c2_start = 5
    c1 = Car(start=(c1_start, NORTH), goal=13, spec='Ga & Fb', actions=actions, world=w, personality=[0, 2, 1, 0, 1])
    c2 = Car(start=(c2_start, SOUTH), goal=22, spec='Ga & Fb', actions=actions, world=w, personality=[0, 2, 1, 0, 1])

    # initialize obstacles
    c1Old = c1_start
    c2Old = c2_start
    w.obsMap[w.cell(c1Old)] = 1
    w.obsMap[w.cell(c2Old)] = 1

    # step machines
    c1Path = [c1Old]
    c2Path = [c2Old]
    for i in range(10):
        print('-----------')
        print('step', i)

        # Car 1 makes a move...
        print('Car 1')
        print(np.rot90(w.obsMap))
        _, _, c1Loc = c1.step(w)
        c1Loc = c1Loc[0]
        w.obsMap[w.cell(c1Old)] = 0
        w.obsMap[w.cell(c1Loc)] = 1
        c1Old = c1Loc
        c1Path.append(c1Loc)
        print('Car 1 decides to go to ', c1Old)
        print()


        # Car 2 makes a move...
        print('Car 2')
        print(np.rot90(w.obsMap))
        _, _, c2Loc = c2.step(w)
        c2Loc = c2Loc[0]
        w.obsMap[w.cell(c2Old)] = 0
        w.obsMap[w.cell(c2Loc)] = 1
        c2Old = c2Loc
        c2Path.append(c2Loc)
        print('Car 2 decides to go to ', c2Old)
        print()

        createImg(w.cell(c1Old), w.cell(c2Old), i)
        input()

    print('C1: Path = ', c1Path)
    print('C2: Path = ', c2Path)



