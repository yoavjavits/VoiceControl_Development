package com.example.voicecontrol;

public class Cube extends GLShape {

    public Cube(GLWorld world, float left, float bottom, float back, float right, float top, float front) {
        super(world);
        GLVertex leftBottomBack = addVertex(left, bottom, back);
        GLVertex rightBottomBack = addVertex(right, bottom, back);
        GLVertex leftTopBack = addVertex(left, top, back);
        GLVertex rightTopBack = addVertex(right, top, back);
        GLVertex leftBottomFront = addVertex(left, bottom, front);
        GLVertex rightBottomFront = addVertex(right, bottom, front);
        GLVertex leftTopFront = addVertex(left, top, front);
        GLVertex rightTopFront = addVertex(right, top, front);
        addFace(new GLFace(leftBottomBack, leftBottomFront, rightBottomFront, rightBottomBack));
        addFace(new GLFace(leftBottomFront, leftTopFront, rightTopFront, rightBottomFront));
        addFace(new GLFace(leftBottomBack, leftTopBack, leftTopFront, leftBottomFront));
        addFace(new GLFace(rightBottomBack, rightBottomFront, rightTopFront, rightTopBack));
        addFace(new GLFace(leftBottomBack, rightBottomBack, rightTopBack, leftTopBack));
        addFace(new GLFace(leftTopBack, rightTopBack, rightTopFront, leftTopFront));

    }

    public static final int kBottom = 0;
    public static final int kFront = 1;
    public static final int kLeft = 2;
    public static final int kRight = 3;
    public static final int kBack = 4;
    public static final int kTop = 5;

}