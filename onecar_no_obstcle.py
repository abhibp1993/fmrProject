from ltlcar import *

if __name__ == '__main__':
    # actions = [(lambda x: tuple([x[0] + 1, x[1]])),  # Right
    #            (lambda x: tuple([x[0], x[1] + 1])),  # Up
    #            (lambda x: tuple([x[0] - 1, x[1]])),  # Left
    #            (lambda x: tuple([x[0], x[1] - 1]))]  # Down

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
    w = World(roadMap='world55/road2.bmp', dim=10, grassMap='world55/grass2.bmp')
    #w.obsMap[0, 1] = 1
    print('---Obs---', '\n', np.rot90(w.obsMap))
    print('---Obs---', '\n', np.rot90(w.roadMap))

    c = Car(start=(9, NORTH), goal=90, spec='Ga & Fb', actions=actions, world=w, personality=[0,9, 0, 0, 0])
    print(c.transduce([w, w, w, w, w, w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w]))


    #image creation portion
    img = Image.new( 'RGB', (w.dim,w.dim), "white") # create a new black image
    pixels = img.load() # create the pixel map
    #find original plan
    c.transduce([w])
    originalPath = c.router.currState
    for x in range(0,w.dim):
        for y in range(0,w.dim):
            pixels[x,y] = (0,0,0)
    for act in range(0, len(originalPath)):
        pos = w.cell(originalPath[act])
        pixels[pos[1],pos[0]] = (0, 0, 255) # set the colour accordingly
    #solve actual

    actualPath = c.transduce([w, w, w, w, w,w,w,w,w,w,w,w,w,w,w,w,w,w,w, w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w])
    print(actualPath)

    #modify image
    for act in range(0, len(actualPath)):
        pos = w.cell(actualPath[act][2][0])
        pixels[pos[1],pos[0]] = (255,0,pixels[pos[1],pos[0]][2]) # set the colour accordingly
    img = img.resize((500, 500))
    img.rotate(270).show()
    img.show()

    image = img.rotate(270)
    image.save('img1.bmp')
    # print(r.transduce([0, 1, 2, 3, 3]))