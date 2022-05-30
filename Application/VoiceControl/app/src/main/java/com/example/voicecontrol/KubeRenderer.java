package com.example.voicecontrol;

import android.opengl.GLSurfaceView;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

class KubeRenderer implements GLSurfaceView.Renderer {
    public interface AnimationCallback {
        void animate();
    }

    public KubeRenderer(GLWorld world, AnimationCallback callback) {
        mWorld = world;
        mCallback = callback;
    }

    public void onDrawFrame(GL10 gl) {
        if (mCallback != null) {
            mCallback.animate();
        }
        gl.glClearColor(0.5f,0.5f,0.5f,1);
        gl.glClear(GL10.GL_COLOR_BUFFER_BIT | GL10.GL_DEPTH_BUFFER_BIT);
        gl.glMatrixMode(GL10.GL_MODELVIEW);
        gl.glLoadIdentity();
        gl.glTranslatef(0, 0, -3.0f);
        gl.glScalef(0.5f, 0.5f, 0.5f);
        gl.glRotatef(mYangle, 1, 0, 0);
        gl.glRotatef(mAngle, 0, 1, 0);

        gl.glColor4f(0.7f, 0.7f, 0.7f, 1.0f);
        gl.glEnableClientState(GL10.GL_VERTEX_ARRAY);
        gl.glEnableClientState(GL10.GL_COLOR_ARRAY);
        gl.glEnable(GL10.GL_CULL_FACE);
        gl.glShadeModel(GL10.GL_SMOOTH);
        gl.glEnable(GL10.GL_DEPTH_TEST);

        mWorld.draw(gl);
    }


    public void onSurfaceChanged(GL10 gl, int width, int height) {
        gl.glViewport(0, 0, width, height);
        float ratio = (float)width / height;
        gl.glMatrixMode(GL10.GL_PROJECTION);
        gl.glLoadIdentity();
        gl.glFrustumf(-ratio, ratio, -1, 1, 2, 12);
        gl.glDisable(GL10.GL_DITHER);
        gl.glActiveTexture(GL10.GL_TEXTURE0);
    }

    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
    }

    public void setAngle(float angle) {
        mAngle = angle;
    }

    public float getAngle() {
        return mAngle;
    }

    public void setYAngle(float angle) {
        mYangle = angle;
    }

    public float getYAngle() {
        return mYangle;
    }

    private GLWorld mWorld;
    private AnimationCallback mCallback;
    private float mAngle;
    private float mYangle;
}