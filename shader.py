"""
core/shader.py
Small helper around raw PyOpenGL shader calls: compiles a vertex+fragment pair,
links them into a program, and raises a readable error (including the GLSL
compiler log) if anything fails. Also provides a thin Shader wrapper with
convenience setters for common uniform types, since typing out
glGetUniformLocation + glUniformX everywhere gets noisy fast.
"""

from OpenGL.GL import (
    glCreateShader, glShaderSource, glCompileShader, glGetShaderiv,
    glGetShaderInfoLog, GL_COMPILE_STATUS, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
    glCreateProgram, glAttachShader, glLinkProgram, glGetProgramiv,
    glGetProgramInfoLog, GL_LINK_STATUS, glDeleteShader, glUseProgram,
    glGetUniformLocation, glUniform1i, glUniform1f, glUniform2f, glUniform3f,
    glUniform4f, glUniformMatrix4fv,
)


class ShaderCompileError(Exception):
    """Raised when a GLSL shader fails to compile or a program fails to link."""
    pass


def _compile_stage(source: str, stage_enum: int, stage_name: str) -> int:
    shader_id = glCreateShader(stage_enum)
    glShaderSource(shader_id, source)
    glCompileShader(shader_id)
    ok = glGetShaderiv(shader_id, GL_COMPILE_STATUS)
    if not ok:
        log = glGetShaderInfoLog(shader_id)
        log_text = log.decode("utf-8", errors="replace") if isinstance(log, bytes) else str(log)
        raise ShaderCompileError(f"{stage_name} shader compile error:\n{log_text}")
    return shader_id


class Shader:
    """A linked GLSL program plus convenience uniform setters."""

    def __init__(self, vertex_src: str, fragment_src: str, name: str = "shader"):
        self.name = name
        vs = _compile_stage(vertex_src, GL_VERTEX_SHADER, f"{name}/vertex")
        fs = _compile_stage(fragment_src, GL_FRAGMENT_SHADER, f"{name}/fragment")

        program = glCreateProgram()
        glAttachShader(program, vs)
        glAttachShader(program, fs)
        glLinkProgram(program)

        ok = glGetProgramiv(program, GL_LINK_STATUS)
        if not ok:
            log = glGetProgramInfoLog(program)
            log_text = log.decode("utf-8", errors="replace") if isinstance(log, bytes) else str(log)
            raise ShaderCompileError(f"{name} link error:\n{log_text}")

        # shader stages can be deleted once linked into the program
        glDeleteShader(vs)
        glDeleteShader(fs)

        self.program = program
        self._uniform_cache = {}

    def use(self):
        glUseProgram(self.program)

    def _loc(self, uniform_name: str) -> int:
        if uniform_name not in self._uniform_cache:
            self._uniform_cache[uniform_name] = glGetUniformLocation(self.program, uniform_name)
        return self._uniform_cache[uniform_name]

    def set_int(self, uniform_name: str, value: int):
        glUniform1i(self._loc(uniform_name), value)

    def set_float(self, uniform_name: str, value: float):
        glUniform1f(self._loc(uniform_name), value)

    def set_vec2(self, uniform_name: str, x: float, y: float):
        glUniform2f(self._loc(uniform_name), x, y)

    def set_vec3(self, uniform_name: str, x: float, y: float, z: float):
        glUniform3f(self._loc(uniform_name), x, y, z)

    def set_vec4(self, uniform_name: str, x: float, y: float, z: float, w: float):
        glUniform4f(self._loc(uniform_name), x, y, z, w)

    def set_mat4(self, uniform_name: str, matrix4x4, transpose: bool = False):
        """matrix4x4: a flat 16-float sequence or numpy array in column-major order."""
        glUniformMatrix4fv(self._loc(uniform_name), 1, transpose, matrix4x4)
