'''this is an offscreen voronoi diagram bitmap renderer for depth completion. 
the output is a (W, H, 4) 8bit numpy array,
we will encode the depth value use the first 2(R, G) channels,
and encode confidence map use the last(B) channel.
A channel is going to be used, so all ones.(due to the nature of renderer)

modified from examples provided by vispy
'''

import numpy as np
from vispy import gloo
from vispy import app
from vispy.util.ptime import time
from vispy.gloo.util import _screenshot
from vispy.gloo import gl
try:
    from utils import utils
except:
    import sys
    sys.path.append("..")
    from utils import utils
# WARNING: doesn't work with Qt4 (update() does not call on_draw()??)
app.use_app('glfw')

# this part of shader copy from
# http://www.labri.fr/perso/nrougier/python-opengl/#gpu-voronoi
vertex = """
    uniform vec2 resolution;
    attribute vec2 center;
    attribute vec3 color;
    attribute float radius;
    varying vec2 v_center;
    varying vec3 v_color;
    varying float v_radius;
    void main()
    {
        v_radius = radius;
        v_center = center;
        v_color  = color;
        gl_PointSize = 2.0 + ceil(2.0*radius);
        gl_Position = vec4(2.0*center/resolution-1.0, 0.0, 1.0);
    } """

fragment = """
    varying vec2 v_center;
    varying vec3 v_color;
    varying float v_radius;
    void main()
    {
        vec2 p = (gl_FragCoord.xy - v_center.xy)/v_radius;
        float z = 1.0 - length(p);
        if (z < 0.0) discard;
        gl_FragDepth = (1.0 - z);
        //gl_FragColor = vec4(v_color, 1);
        // use 'z' as confidence
        gl_FragColor = vec4(z, z, z, 1);
        gl_FragColor = vec4(v_color.x, v_color.y, z, 1);
    } """

class Canvas(app.Canvas):
    '''one time renderer, maybe can be extended to a service
    '''

    def __init__(self, size, num_points:int, center, color, radius):
        '''[summary]
        
        Arguments:
            size {[type]} -- size of the rendered image
            num_points {int} -- number of valid depth values
            center {[type]} -- the (x, y) coordinate of each valid depth value
            color {[type]} -- encode the 16bit depth value to (R, G) channels, corresponding to 'center'
            radius -- the radius of circle/cone to render the voronoi diagram
        '''

        # We hide the canvas upon creation.
        app.Canvas.__init__(self, show=False, size=size)
        self._t0 = time()
        # Texture where we render the scene.
        self._rendertex = gloo.Texture2D(shape=size[::-1] + (4,))
        # FBO.
        self._fbo = gloo.FrameBuffer(self._rendertex,
                                     gloo.RenderBuffer(size[::-1]))
        # Regular program that will be rendered to the FBO.

        V = np.zeros(num_points, [("center", np.float32, 2),
                    ("color",  np.float32, 3),
                    ("radius", np.float32, 1)])
        V["center"] = center
        V["color"] = color
        V["radius"] = radius

        self.program = gloo.Program(vertex, fragment)
        self.program.bind(gloo.VertexBuffer(V))
        self.program['resolution'] = self.size
        # We manually draw the hidden canvas.
        self.update()


    def on_draw(self, event):
        # Render in the FBO.
        with self._fbo:
            gl.glEnable(gl.GL_DEPTH_TEST)
            gloo.clear('black')
            gloo.set_viewport(0, 0, *self.size)
            self.program.draw(gl.GL_POINTS)
            # Retrieve the contents of the FBO texture.
            self.im = _screenshot((0, 0, self.size[0], self.size[1]))
        self._time = time() - self._t0
        # Immediately exit the application.
        app.quit()
