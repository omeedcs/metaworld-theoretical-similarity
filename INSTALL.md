# Installing Stuff on the M1

## Meta-World
Following [this guide](https://github.com/openai/mujoco-py/issues/662#issuecomment-996081734) got us most of the way there, although there was one last issue that needed to be resolved:

```
❯ python3 -c 'import mujoco_py'                                                                                                                                                                                 (meta-world)
Import error. Trying to rebuild mujoco_py.
running build_ext
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/mujoco_py/__init__.py", line 2, in <module>
    from mujoco_py.builder import cymj, ignore_mujoco_warnings, functions, MujocoException
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/mujoco_py/builder.py", line 504, in <module>
    cymj = load_cython_ext(mujoco_path)
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/mujoco_py/builder.py", line 111, in load_cython_ext
    mod = load_dynamic_ext('cymj', cext_so_path)
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/mujoco_py/builder.py", line 130, in load_dynamic_ext
    return loader.load_module()
ImportError: dlopen(/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/mujoco_py/generated/cymj_2.1.2.14_310_macextensionbuilder_310.so, 0x0002): Library not loaded: @rpath/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib
  Referenced from: <DD6C1BDD-BE34-3A9F-BFE2-C7B5BBD411AD> /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/mujoco_py/generated/cymj_2.1.2.14_310_macextensionbuilder_310.so
  Reason: tried: '/Users/arpandhatt/.mujoco/mujoco210/bin/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Users/arpandhatt/.mujoco/mujoco210/bin/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/mujoco_py/generated/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/gcc/aarch64-apple-darwin22/11/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/gcc/aarch64-apple-darwin22/11/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/gcc/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/gcc/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/Users/arpandhatt/.mujoco/mujoco210/bin/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Users/arpandhatt/.mujoco/mujoco210/bin/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/mujoco_py/generated/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/gcc/aarch64-apple-darwin22/11/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/gcc/aarch64-apple-darwin22/11/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/gcc/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/gcc/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/Sy
stem/Volumes/Preboot/Cryptexes/OS/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS@rpath/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/Users/arpandhatt/.mujoco/mujoco210/bin/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Users/arpandhatt/.mujoco/mujoco210/bin/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/mujoco_py/generated/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/gcc/aarch64-apple-darwin22/11/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/gcc/aarch64-apple-darwin22/11/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/gcc/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/gcc/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/Users/arpandhatt/.mujoco/mujoco210/bin/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Users/arpandhatt/.mujoco/mujoco210/bin/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/mujoco_py/generated/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/gcc/aarch64-apple-darwin22/11/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/gcc/aarch64-apple-darwin22/11/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/gcc/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/gcc/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/opt/homebrew/Cellar/gcc@11/11.3.0/lib/gcc/11/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib' (no such file), '/usr/lib/libmujoco.2.1.1.dylib' (no such file, not in dyld cache)
```

This is presumably an issue with linking to the `libmujoco` dynamic library. We can fix this by just symlinking the existing `MuJoCo.framework` into one of the directories that it searches for it in:

```bash
cd ~/.mujoco/mujoco210/bin
ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework
```

Try running the `test-mw-installation.py`. There might be some warnings about casting floats, but it shouldn't error. Presumably it's working.

## Stable Baselines
Following [their instructions](https://github.com/DLR-RM/stable-baselines3) worked really well.

### PyTorch
Install Pytorch using the command on the PyTorch website. There shouldn't be any issues.

### PyGLet
OpenAI Gym needs PyGLet to render the cartpole task. Use `pip install pyglet==1.5.27` to get the right version.

Try running `stable-baselines-installation.py` to check if it worked.
