package com.example.voicecontrol;

import java.util.Random;

import android.app.Activity;
import android.graphics.Color;
import android.graphics.drawable.Drawable;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.view.Window;
import android.widget.Button;
import android.widget.GridLayout;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;

import androidx.annotation.NonNull;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

public class MainActivity extends Activity implements KubeRenderer.AnimationCallback {

    private GLWorld makeGLWorld()
    {
        GLWorld world = new GLWorld();

        int one = 0x10000;
        int half = 0x08000;
        GLColor red = new GLColor(one, 0, 0);
        GLColor green = new GLColor(0, one, 0);
        GLColor blue = new GLColor(0, 0, one);
        GLColor yellow = new GLColor(one, one, 0);
        GLColor orange = new GLColor(one, half, 0);
        GLColor white = new GLColor(one, one, one);
        GLColor black = new GLColor(0, 0, 0);

        float c0 = -1.0f;
        float c1 = -0.38f;
        float c2 = -0.32f;
        float c3 = 0.32f;
        float c4 = 0.38f;
        float c5 = 1.0f;

// top back, left to right
        mCubes[0] = new Cube(world, c0, c4, c0, c1, c5, c1);
        mCubes[1] = new Cube(world, c2, c4, c0, c3, c5, c1);
        mCubes[2] = new Cube(world, c4, c4, c0, c5, c5, c1);
// top middle, left to right
        mCubes[3] = new Cube(world, c0, c4, c2, c1, c5, c3);
        mCubes[4] = new Cube(world, c2, c4, c2, c3, c5, c3);
        mCubes[5] = new Cube(world, c4, c4, c2, c5, c5, c3);
// top front, left to right
        mCubes[6] = new Cube(world, c0, c4, c4, c1, c5, c5);
        mCubes[7] = new Cube(world, c2, c4, c4, c3, c5, c5);
        mCubes[8] = new Cube(world, c4, c4, c4, c5, c5, c5);
// middle back, left to right
        mCubes[9] = new Cube(world, c0, c2, c0, c1, c3, c1);
        mCubes[10] = new Cube(world, c2, c2, c0, c3, c3, c1);
        mCubes[11] = new Cube(world, c4, c2, c0, c5, c3, c1);
// middle middle, left to right
        mCubes[12] = new Cube(world, c0, c2, c2, c1, c3, c3);
        mCubes[13] = null;
        mCubes[14] = new Cube(world, c4, c2, c2, c5, c3, c3);
// middle front, left to right
        mCubes[15] = new Cube(world, c0, c2, c4, c1, c3, c5);
        mCubes[16] = new Cube(world, c2, c2, c4, c3, c3, c5);
        mCubes[17] = new Cube(world, c4, c2, c4, c5, c3, c5);
// bottom back, left to right
        mCubes[18] = new Cube(world, c0, c0, c0, c1, c1, c1);
        mCubes[19] = new Cube(world, c2, c0, c0, c3, c1, c1);
        mCubes[20] = new Cube(world, c4, c0, c0, c5, c1, c1);
// bottom middle, left to right
        mCubes[21] = new Cube(world, c0, c0, c2, c1, c1, c3);
        mCubes[22] = new Cube(world, c2, c0, c2, c3, c1, c3);
        mCubes[23] = new Cube(world, c4, c0, c2, c5, c1, c3);
// bottom front, left to right
        mCubes[24] = new Cube(world, c0, c0, c4, c1, c1, c5);
        mCubes[25] = new Cube(world, c2, c0, c4, c3, c1, c5);
        mCubes[26] = new Cube(world, c4, c0, c4, c5, c1, c5);

// paint the sides
        int i, j;
// set all faces black by default
        for (i = 0; i < 27; i++) {
            Cube cube = mCubes[i];
            if (cube != null) {
                for (j = 0; j < 6; j++)
                    cube.setFaceColor(j, black);
            }
        }

// paint top
        for (i = 0; i < 9; i++)
            mCubes[i].setFaceColor(Cube.kTop, orange);
// paint bottom
        for (i = 18; i < 27; i++)
            mCubes[i].setFaceColor(Cube.kBottom, red);
// paint left
        for (i = 0; i < 27; i += 3)
            mCubes[i].setFaceColor(Cube.kLeft, yellow);
// paint right
        for (i = 2; i < 27; i += 3)
            mCubes[i].setFaceColor(Cube.kRight, white);
// paint back
        for (i = 0; i < 27; i += 9)
            for (j = 0; j < 3; j++)
                mCubes[i + j].setFaceColor(Cube.kBack, blue);
// paint front
        for (i = 6; i < 27; i += 9)
            for (j = 0; j < 3; j++)
                mCubes[i + j].setFaceColor(Cube.kFront, green);

        for (i = 0; i < 27; i++)
            if (mCubes[i] != null)
                world.addShape(mCubes[i]);

// initialize our permutation to solved position
        mPermutation = new int[27];
        for (i = 0; i < mPermutation.length; i++)
            mPermutation[i] = i;

        createLayers();
        updateLayers();

        world.generate();

        return world;
    }

    private void createLayers() {
        mLayers[kUp] = new Layer(Layer.kAxisY);
        mLayers[kDown] = new Layer(Layer.kAxisY);
        mLayers[kLeft] = new Layer(Layer.kAxisX);
        mLayers[kRight] = new Layer(Layer.kAxisX);
        mLayers[kFront] = new Layer(Layer.kAxisZ);
        mLayers[kBack] = new Layer(Layer.kAxisZ);
        mLayers[kMiddle] = new Layer(Layer.kAxisX);
        mLayers[kEquator] = new Layer(Layer.kAxisY);
        mLayers[kSide] = new Layer(Layer.kAxisZ);
    }

    private void updateLayers() {
        Layer layer;
        GLShape[] shapes;
        int i, j, k;

// up layer
        layer = mLayers[kUp];
        shapes = layer.mShapes;
        for (i = 0; i < 9; i++)
            shapes[i] = mCubes[mPermutation[i]];

// down layer
        layer = mLayers[kDown];
        shapes = layer.mShapes;
        for (i = 18, k = 0; i < 27; i++)
            shapes[k++] = mCubes[mPermutation[i]];

// left layer
        layer = mLayers[kLeft];
        shapes = layer.mShapes;
        for (i = 0, k = 0; i < 27; i += 9)
            for (j = 0; j < 9; j += 3)
                shapes[k++] = mCubes[mPermutation[i + j]];

// right layer
        layer = mLayers[kRight];
        shapes = layer.mShapes;
        for (i = 2, k = 0; i < 27; i += 9)
            for (j = 0; j < 9; j += 3)
                shapes[k++] = mCubes[mPermutation[i + j]];

// front layer
        layer = mLayers[kFront];
        shapes = layer.mShapes;
        for (i = 6, k = 0; i < 27; i += 9)
            for (j = 0; j < 3; j++)
                shapes[k++] = mCubes[mPermutation[i + j]];

// back layer
        layer = mLayers[kBack];
        shapes = layer.mShapes;
        for (i = 0, k = 0; i < 27; i += 9)
            for (j = 0; j < 3; j++)
                shapes[k++] = mCubes[mPermutation[i + j]];

// middle layer
        layer = mLayers[kMiddle];
        shapes = layer.mShapes;
        for (i = 1, k = 0; i < 27; i += 9)
            for (j = 0; j < 9; j += 3)
                shapes[k++] = mCubes[mPermutation[i + j]];

// equator layer
        layer = mLayers[kEquator];
        shapes = layer.mShapes;
        for (i = 9, k = 0; i < 18; i++)
            shapes[k++] = mCubes[mPermutation[i]];

// side layer
        layer = mLayers[kSide];
        shapes = layer.mShapes;
        for (i = 3, k = 0; i < 27; i += 9)
            for (j = 0; j < 3; j++)
                shapes[k++] = mCubes[mPermutation[i + j]];
    }

    public void rotate_function(int rotate_index, int delay){
        if (mCurrentLayer == null) {
            //mRandom.nextInt(9);
            mCurrentLayer = mLayers[rotate_index];
            mCurrentLayerPermutation = mLayerPermutations[rotate_index];
            mCurrentLayer.startAnimation();
            boolean direction = mRandom.nextBoolean();
            int count = mRandom.nextInt(3) + 1;

            count = 1;
            direction = false;
            mCurrentAngle = 0;
            if (direction) {
                mAngleIncrement = (float)Math.PI / 50;
                mEndAngle = mCurrentAngle + ((float)Math.PI * count) / 2f;
            } else {
                mAngleIncrement = -(float)Math.PI / 50;
                mEndAngle = mCurrentAngle - ((float)Math.PI * count) / 2f;
            }
        }

        while (!((mAngleIncrement > 0f && mCurrentAngle >= mEndAngle) ||
                (mAngleIncrement < 0f && mCurrentAngle <= mEndAngle))) {
            mCurrentAngle += mAngleIncrement;
            mCurrentLayer.setAngle(mCurrentAngle);
            try {
                Thread.sleep(delay);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        if ((mAngleIncrement > 0f && mCurrentAngle >= mEndAngle) ||
                (mAngleIncrement < 0f && mCurrentAngle <= mEndAngle)) {
            mCurrentLayer.setAngle(mEndAngle);
            mCurrentLayer.endAnimation();
            mCurrentLayer = null;

// adjust mPermutation based on the completed layer rotation
            int[] newPermutation = new int[27];
            for (int i = 0; i < 27; i++) {
                newPermutation[i] = mPermutation[mCurrentLayerPermutation[i]];
            }
            mPermutation = newPermutation;
            updateLayers();

        } else {
            mCurrentLayer.setAngle(mCurrentAngle);
        }
    }

    int convert_to_k(String direction){
        switch (direction) {
            case "up" : return kUp;
            case "down" : return kDown;
            case "left" : return kLeft;
            case "right" : return kRight;
            case "backward" : return kBack;
            case "forward" : return kFront;
            default: return 0;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);

// We don't need a title either.
        requestWindowFeature(Window.FEATURE_NO_TITLE);

        mView = new GLSurfaceView(getApplication());
        //mView = (GLSurfaceView) findViewById(R.id.gl);
        mRenderer = new KubeRenderer(makeGLWorld(), this);
        mView.setRenderer(mRenderer);
        setContentView(mView);


        GridLayout rotate_ll = new GridLayout(this);

        Button rotate_right = new Button(this);
        Button rotate_left = new Button(this);
        Button rotate_up = new Button(this);
        Button rotate_down = new Button(this);
        Button rotate_backward = new Button(this);
        Button rotate_forward = new Button(this);
        Button scramble = new Button(this);


        rotate_right.setText("R");
        rotate_right.setTextColor(Color.parseColor("#ffffff"));
        rotate_left.setText("L");
        rotate_left.setTextColor(Color.parseColor("#ffffff"));
        rotate_up.setText("U");
        rotate_up.setTextColor(Color.parseColor("#ffffff"));
        rotate_down.setText("D");
        rotate_down.setTextAlignment(View.TEXT_ALIGNMENT_CENTER);
        rotate_down.setTextColor(Color.parseColor("#ffffff"));
        rotate_backward.setText("B");
        rotate_backward.setTextAlignment(View.TEXT_ALIGNMENT_CENTER);
        rotate_backward.setTextColor(Color.parseColor("#ffffff"));
        rotate_forward.setText("F");
        rotate_forward.setTextColor(Color.parseColor("#ffffff"));
        rotate_forward.setTextAlignment(View.TEXT_ALIGNMENT_CENTER);
        scramble.setText("Scramble");

        rotate_down.setGravity(Gravity.BOTTOM);
        rotate_backward.setGravity(Gravity.BOTTOM);
        rotate_forward.setGravity(Gravity.BOTTOM);
        scramble.setGravity(Gravity.CENTER_HORIZONTAL);

        rotate_ll.addView(rotate_right);
        rotate_ll.addView(rotate_left);
        rotate_ll.addView(rotate_up);
        rotate_ll.addView(rotate_down);
        rotate_ll.addView(rotate_forward);
        rotate_ll.addView(rotate_backward);
        rotate_ll.addView(scramble);

        //rotate_ll.setAlignmentMode(GridLayout.ALIGN_MARGINS);
        rotate_ll.setOrientation(GridLayout.HORIZONTAL);
        rotate_ll.setColumnCount(3);
        rotate_ll.setRowCount(3);


        //rotate_ll.setGravity(Gravity.TOP | Gravity.CENTER_HORIZONTAL);
        this.addContentView(rotate_ll,
                new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.MATCH_PARENT));

        rotate_right.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                rotate_function(kRight, 18);
                rotate.setValue("right");
            }
        });

        rotate_left.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                rotate_function(kLeft, 18);
                rotate.setValue("left");
            }
        });
        rotate_up.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                rotate_function(kUp, 18);
                rotate.setValue("up");
            }
        });
        rotate_down.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                rotate_function(kDown, 18);
                rotate.setValue("down");

            }
        });
        rotate_forward.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                rotate_function(kFront, 18);
                rotate.setValue("front");
            }
        });
        rotate_backward.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                rotate_function(kBack, 18);
                rotate.setValue("back");
            }
        });
        scramble.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                for (int i = 0 ; i < 30 ; i ++){
                    int random = mRandom.nextInt(9);
                    rotate_function(random,5);
                }
            }
        });

        //rotate.setValue("up");
        rotate.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                String direction = snapshot.getValue(String.class);
                int dir_k = convert_to_k(direction);
                rotate_function(dir_k,18);
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {

            }
        });


        //turn the cube buttons
        LinearLayout ll = new LinearLayout(this);
        Button right = new Button(this);
        Button left = new Button(this);
        Button down = new Button(this);
        Button up = new Button(this);
        left.setText("left");
        right.setText("right");
        down.setText("down");
        up.setText("up");
        ll.addView(right);
        ll.addView(left);
        ll.addView(down);
        ll.addView(up);
        ll.setGravity(Gravity.BOTTOM | Gravity.CENTER_HORIZONTAL);

        this.addContentView(ll,
                new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.MATCH_PARENT));


        right.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mRenderer.setAngle(mRenderer.getAngle() + 10f);
            }
        });
        left.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mRenderer.setAngle(mRenderer.getAngle() - 10f);
            }
        });
        down.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mRenderer.setYAngle(mRenderer.getYAngle() + 10f);
            }
        });
        up.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mRenderer.setYAngle(mRenderer.getYAngle() - 10f);

            }
        });
    }



    @Override
    protected void onResume()
    {
        super.onResume();
        mView.onResume();
    }

    @Override
    protected void onPause()
    {
        super.onPause();
        mView.onPause();
    }





    public void animate() {

    }

    GLSurfaceView mView;
    KubeRenderer mRenderer;
    Cube[] mCubes = new Cube[27];
    // a Layer for each possible move
    Layer[] mLayers = new Layer[9];
    // permutations corresponding to a pi/2 rotation of each layer about its axis
    static int[][] mLayerPermutations = {
// permutation for UP layer
            { 2, 5, 8, 1, 4, 7, 0, 3, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26 },
// permutation for DOWN layer
            { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 23, 26, 19, 22, 25, 18, 21, 24 },
// permutation for LEFT layer
            { 6, 1, 2, 15, 4, 5, 24, 7, 8, 3, 10, 11, 12, 13, 14, 21, 16, 17, 0, 19, 20, 9, 22, 23, 18, 25, 26 },
// permutation for RIGHT layer
            { 0, 1, 8, 3, 4, 17, 6, 7, 26, 9, 10, 5, 12, 13, 14, 15, 16, 23, 18, 19, 2, 21, 22, 11, 24, 25, 20 },
// permutation for FRONT layer
            { 0, 1, 2, 3, 4, 5, 24, 15, 6, 9, 10, 11, 12, 13, 14, 25, 16, 7, 18, 19, 20, 21, 22, 23, 26, 17, 8 },
// permutation for BACK layer
            { 18, 9, 0, 3, 4, 5, 6, 7, 8, 19, 10, 1, 12, 13, 14, 15, 16, 17, 20, 11, 2, 21, 22, 23, 24, 25, 26 },
// permutation for MIDDLE layer
            { 0, 7, 2, 3, 16, 5, 6, 25, 8, 9, 4, 11, 12, 13, 14, 15, 22, 17, 18, 1, 20, 21, 10, 23, 24, 19, 26 },
// permutation for EQUATOR layer
            { 0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 14, 17, 10, 13, 16, 9, 12, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26 },
// permutation for SIDE layer
            { 0, 1, 2, 21, 12, 3, 6, 7, 8, 9, 10, 11, 22, 13, 4, 15, 16, 17, 18, 19, 20, 23, 14, 5, 24, 25, 26 }
    };

    // current permutation of starting position
    int[] mPermutation;

    int i = 1;
    // for random cube movements
    Random mRandom = new Random(System.currentTimeMillis());
    // currently turning layer
    Layer mCurrentLayer = null;
    // current and final angle for current Layer animation
    float mCurrentAngle, mEndAngle;
    // amount to increment angle
    float mAngleIncrement;
    int[] mCurrentLayerPermutation;

    // names for our 9 layers (based on notation from http://www.cubefreak.net/notation.html)
    static final int kUp = 0;
    static final int kDown = 1;
    static final int kLeft = 2;
    static final int kRight = 3;
    static final int kFront = 4;
    static final int kBack = 5;
    static final int kMiddle = 6;
    static final int kEquator = 7;
    static final int kSide = 8;

    //fire base

    FirebaseDatabase database = FirebaseDatabase.getInstance();
    DatabaseReference rotate = database.getReference("Rotation/direction");

}