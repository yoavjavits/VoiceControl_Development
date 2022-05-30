import numpy as np


class Cube():
    """ The Cube object consists of the cubestring and the manipulation methods """
    cubestring = str('')
    colors = {}

    def __init__(self):
        self.cubestring = 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB'
        self.colors = {'U': "white", 'R': "red", 'F': "green",
                       'D': "yellow", 'L': "orange", 'B': "blue"}
        pass

    def scramble(self, scramblestring):
        """ scrambles the cubestring based on an input string """
        if len(scramblestring) == 0:
            # no input given --> generate random cube
            pass
        elif len(scramblestring) == 54 and checkString(scramblestring):
            # A cubestring was used
            self.cubestring = scramblestring
        else:
            # A sequence was chosen
            bSequence = True
            switcher = {
                "U": self.U,
                "U'": self.u,
                "R": self.R,
                "R'": self.r,
                "F": self.F,
                "F'": self.f,
                "D": self.D,
                "D'": self.d,
                "L": self.L,
                "L'": self.l,
                "B": self.B,
                "B'": self.b,
                "x": self.rotate_x,
                "x'": self.rotate_xprime,
                "y": self.rotate_y,
                "y'": self.rotate_yprime,
                "z": self.rotate_z,
                "z'": self.rotate_zprime
            }

            # replace 2s in string with preceding letter
            while scramblestring.find("2") != -1:
                index = scramblestring.find("2")
                scramblestring = scramblestring.replace(
                    scramblestring[index], " " + scramblestring[index-1], 1)

            moveset = scramblestring.split(" ")
            for move in moveset:
                func = switcher.get(move, lambda: None)
                func()
        pass

    def genRandom(self):
        """ generates a random cube scramble, for now fixed lengh to 20 moves"""
        import random
        scramble_length = 20
        moves = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2",
                 "D", "D'", "D2", "L", "L'", "L2", "B", "B'", "B2"]
        random_moves = []
        for index in range(0, scramble_length):
            currmove = moves[random.randint(0, len(moves) - 1)]
            random_moves.append(currmove)
            random.shuffle(random_moves)
        random_string = " ".join(random_moves)

        return random_string

    def getCubeString(self, newfaces):
        """ Takes the cubefaces and and return the cubestring """
        newcubestring = ''
        order = ["U", "R", "F", "D", "L", "B"]
        for face in order:
            # transform each array into 1x9
            newfaces[face] = np.reshape(newfaces[face], (1, 9))
            # convert back to bytes and decode
            newfaces[face] = newfaces[face].tobytes().decode()
            # concatenate the strings
            newcubestring = newcubestring+newfaces[face]
        return newcubestring

    def getCubeFaces(self):
        """ Takes the cubestring and returns a dictionary with numpy arrays """
        oldcubestring_bytes = str.encode(self.cubestring)
        # create dict with the faces and the starting indices in the cubestring
        oldfaces = {'U': 0, 'R': 9, 'F': 18, 'D': 27, 'L': 36, 'B': 45}
        newfaces = dict()
        for face in oldfaces:
            # transform string into 6 1x9 arrays
            oldfaces[face] = np.frombuffer(
                oldcubestring_bytes, dtype="S1", count=9, offset=oldfaces[face])
            # transform each array into 3x3
            oldfaces[face] = np.reshape(oldfaces[face], (3, 3))
            # needs to be copied using numpy because just copying the dict makes the array in newfaces read-only
            newfaces[face] = np.copy(oldfaces[face])
        return oldfaces, newfaces

    def U(self):
        """Turn the upper layer clockwise"""
        oldfaces, newfaces = self.getCubeFaces()

        newfaces['U'] = np.rot90(oldfaces['U'], k=-1)
        newfaces['L'][0] = oldfaces['F'][0]
        newfaces['F'][0] = oldfaces['R'][0]
        newfaces['R'][0] = oldfaces['B'][0]
        newfaces['B'][0] = oldfaces['L'][0]

        newcubestring = self.getCubeString(newfaces)
        self.cubestring = newcubestring
        pass

    def u(self):
        """Turn the upper layer counterclockwise.
            rotate up face counter-clockwise
            upper 3 cubies of F,L,R,B are rotated in order"""

        oldfaces, newfaces = self.getCubeFaces()

        newfaces['U'] = np.rot90(oldfaces['U'], k=1)
        newfaces['F'][0] = oldfaces['L'][0]
        newfaces['R'][0] = oldfaces['F'][0]
        newfaces['B'][0] = oldfaces['R'][0]
        newfaces['L'][0] = oldfaces['B'][0]

        newcubestring = self.getCubeString(newfaces)

        self.cubestring = newcubestring
        pass

    def R(self):
        """Turn right layer clockwise by flipping to the top rotating there and flipping back"""
        self.rotate_zprime()
        self.U()
        self.rotate_z()
        pass

    def r(self):
        """Turn right layer counter-clockwise by flipping to the top rotating there and flipping back"""
        self.rotate_zprime()
        self.u()
        self.rotate_z()
        pass

    def F(self):
        """Turn front layer clockwise by flipping to the top rotating there and flipping back"""
        self.rotate_x()
        self.U()
        self.rotate_xprime()
        pass

    def f(self):
        """Turn front layer counter-clockwise by flipping to the top rotating there and flipping back"""
        self.rotate_x()
        self.u()
        self.rotate_xprime()
        pass

    def D(self):
        """Turn down layer clockwise by flipping to the top rotating there and flipping back"""
        self.rotate_xprime()
        self.rotate_xprime()
        self.U()
        self.rotate_x()
        self.rotate_x()
        pass

    def d(self):
        """Turn down layer counter-clockwise by flipping to the top rotating there and flipping back"""
        self.rotate_xprime()
        self.rotate_xprime()
        self.u()
        self.rotate_x()
        self.rotate_x()
        pass

    def L(self):
        """Turn left layer clockwise by flipping to the top rotating there and flipping back"""
        self.rotate_z()
        self.U()
        self.rotate_zprime()
        pass

    def l(self):
        """Turn left layer counter-clockwise by flipping to the top rotating there and flipping back"""
        self.rotate_z()
        self.u()
        self.rotate_zprime()
        pass

    def B(self):
        """Turn back layer clockwise by flipping to the top rotating there and flipping back"""
        self.rotate_xprime()
        self.U()
        self.rotate_x()
        pass

    def b(self):
        """Turn back layer counter-clockwise by flipping to the top rotating there and flipping back"""
        self.rotate_xprime()
        self.u()
        self.rotate_x()
        pass

    def rotate_y(self):
        """Rearrange the cubefaces to emulate the rotation of the cube in clockwise y-direction
            Imagine an axis through U/D faces """
        oldfaces, newfaces = self.getCubeFaces()
        newfaces['U'] = np.rot90(oldfaces['U'], k=-1)
        newfaces['D'] = np.rot90(oldfaces['D'], k=1)
        newfaces['L'] = oldfaces['F']
        newfaces['F'] = oldfaces['R']
        newfaces['R'] = oldfaces['B']
        newfaces['B'] = oldfaces['L']

        newcubestring = self.getCubeString(newfaces)

        self.cubestring = newcubestring
        pass

    def rotate_yprime(self):
        """Rearrange the cubefaces to emulate the rotation of the cube in counterclockwise y-direction """
        oldfaces, newfaces = self.getCubeFaces()
        newfaces['U'] = np.rot90(oldfaces['U'], k=1)
        newfaces['D'] = np.rot90(oldfaces['D'], k=-1)
        newfaces['F'] = oldfaces['L']
        newfaces['R'] = oldfaces['F']
        newfaces['B'] = oldfaces['R']
        newfaces['L'] = oldfaces['B']

        newcubestring = self.getCubeString(newfaces)

        self.cubestring = newcubestring
        pass

    def rotate_x(self):
        """Rearrange the cubefaces to emulate the rotation of the cube in clockwise x-direction
            Imagine axis through L/R faces"""
        oldfaces, newfaces = self.getCubeFaces()

        newfaces['R'] = np.rot90(oldfaces['R'], k=-1)
        newfaces['L'] = np.rot90(oldfaces['L'], k=1)
        oldfaces['U'] = np.rot90(oldfaces['U'], k=-2)
        oldfaces['B'] = np.rot90(oldfaces['B'], k=-2)
        newfaces['B'] = oldfaces['U']
        newfaces['U'] = oldfaces['F']
        newfaces['F'] = oldfaces['D']
        newfaces['D'] = oldfaces['B']

        newcubestring = self.getCubeString(newfaces)

        self.cubestring = newcubestring
        pass

    def rotate_xprime(self):
        """Rearrange the cubefaces to emulate the rotation of the cube in clockwise x-direction"""
        oldfaces, newfaces = self.getCubeFaces()

        newfaces['R'] = np.rot90(oldfaces['R'], k=1)
        newfaces['L'] = np.rot90(oldfaces['L'], k=-1)
        oldfaces['D'] = np.rot90(oldfaces['D'], k=2)
        oldfaces['B'] = np.rot90(oldfaces['B'], k=2)
        newfaces['B'] = oldfaces['D']
        newfaces['U'] = oldfaces['B']
        newfaces['F'] = oldfaces['U']
        newfaces['D'] = oldfaces['F']

        newcubestring = self.getCubeString(newfaces)

        self.cubestring = newcubestring
        pass

    def rotate_z(self):
        """Rearrange the cubefaces to emulate the rotation of the cube in clockwise x-direction
            Imagine axis through F/B faces"""
        oldfaces, newfaces = self.getCubeFaces()
        oldfaces['U'] = np.rot90(oldfaces['U'], k=-1)
        oldfaces['F'] = np.rot90(oldfaces['F'], k=-1)
        oldfaces['R'] = np.rot90(oldfaces['R'], k=-1)
        oldfaces['D'] = np.rot90(oldfaces['D'], k=-1)
        oldfaces['L'] = np.rot90(oldfaces['L'], k=-1)
        oldfaces['B'] = np.rot90(oldfaces['B'], k=1)

        newfaces['F'] = oldfaces['F']
        newfaces['B'] = oldfaces['B']
        newfaces['R'] = oldfaces['U']
        newfaces['D'] = oldfaces['R']
        newfaces['L'] = oldfaces['D']
        newfaces['U'] = oldfaces['L']

        newcubestring = self.getCubeString(newfaces)

        self.cubestring = newcubestring
        pass

    def rotate_zprime(self):
        """Rearrange the cubefaces to emulate the rotation of the cube in clockwise x-direction"""
        oldfaces, newfaces = self.getCubeFaces()

        oldfaces['U'] = np.rot90(oldfaces['U'], k=1)
        oldfaces['F'] = np.rot90(oldfaces['F'], k=1)
        oldfaces['R'] = np.rot90(oldfaces['R'], k=1)
        oldfaces['D'] = np.rot90(oldfaces['D'], k=1)
        oldfaces['L'] = np.rot90(oldfaces['L'], k=1)
        oldfaces['B'] = np.rot90(oldfaces['B'], k=-1)

        newfaces['F'] = oldfaces['F']
        newfaces['B'] = oldfaces['B']
        newfaces['R'] = oldfaces['D']
        newfaces['D'] = oldfaces['L']
        newfaces['L'] = oldfaces['U']
        newfaces['U'] = oldfaces['R']

        newcubestring = self.getCubeString(newfaces)

        self.cubestring = newcubestring
        pass

    def sendCommands(self, scramblestring):
        """Send a string of instructions to the hardware"""
        validCMDs = {"U": "U1",  # standard moves
                     "U'": "u1",
                     "U2": "U2",
                     "R": "R1",
                     "R'": "r1",
                     "R2": "R2",
                     "F": "F1",
                     "F'": "f1",
                     "F2": "F2",
                     "D": "D1",
                     "D'": "d1",
                     "D2": "D2",
                     "L": "L1",
                     "L'": "l1",
                     "L2": "L2",
                     "B": "B1",
                     "B'": "b1",
                     "B2": "B2",
                     "Y0": "Y0",  # axis open/close
                     "y0": "y0",
                     "X0": "X0",
                     "x0": "x0",
                     "E1": "E1",  # motor en/disable
                     "E2": "E2",
                     "A1": "A1",  # single steps
                     "a1": "a1",
                     "A2": "A2",
                     "a2": "a2",
                     "A3": "A3",
                     "a3": "a3",
                     "A4": "A4",
                     "a4": "a4",
                     "A5": "A5",
                     "a5": "a5",
                     "A6": "A6",
                     "a6": "a6",
                     }

        txCMDs = []
        txList = scramblestring.split(" ")
        for CMD in txList:
            CMD = validCMDs.get(CMD, lambda: "")
            txCMDs.append(CMD)

        for num in txCMDs:
            # TODO send command to serial port
            pass

        return 0

    def generateSolution(self):
        """Generate a solution for the current cubestring and display in the GUI"""
        solution = koci.solve(self.cube.cubestring)
        return solution


def checkString(strg):
    """ checks if the input string contains 
        returns True if it only contains the characters URFDLB """
    import re
    search = re.compile(r'[^URFDLB]').search
    result = bool(search(strg))
    return not result
