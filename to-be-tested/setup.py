#!/usr/bin/env python


from distutils.core import Command, setup
from distutils.dep_util import newer, newer_group
from distutils.command.build import build as _build
from distutils import log
import os
import sys



data_files = []
extra = {}

# check if we are on windows
if os.name == 'nt':
    import py2exe
    from glob import glob
    
    # py2exe stuff - lifted from old shamba gui (0.7) in misc/v1.0gui
    data_files.append(("Microsoft.VC90.CRT", glob(r'C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\redist\x86\Microsoft.VC90.CRT\*.*')))
    sys.path.append("C:\\Program Files (x86)\\Microsoft Visual Studio 9.0\\VC\\redist\\x86\\Microsoft.VC90.CRT")

    # extra options to include in the setup
    extra['windows'] = [{"script":"shamba.pyw"}]
    extra['options'] = {
            "py2exe" : {
                   "skip_archive" : 0,

                    # Don't pull in all this MFC stuff used by the makepy UI.
                    "excludes" : "pywin,pywin.dialogs,pywin.dialogs.list"
                    ",setup,distutils",  # required only for in-place use
                    "includes" : ["sip"],
                    "optimize" : 1
        }
    }


class build_qt(Command):
    """lifted from https://bitbucket.org/tortoisehg/thg/src/11df59e9cfbe/setup.py"""
    description = "build PyQt GUIs (.ui) and resources (.qrc)"
    user_options = [('force', 'f', 'forcibly compile everything'
                     ' (ignore file timestamps)'),
                    ('frozen', None, 'include resources for frozen exe')]
    boolean_options = ('force', 'frozen')

    def initialize_options(self):
        self.force = None
        self.frozen = False

    def finalize_options(self):
        self.set_undefined_options('build', ('force', 'force'))

    def compile_ui(self, ui_file, py_file=None):
        # Search for pyuic4 in python bin dir, then in the $Path.
        if py_file is None:
            py_file = os.path.splitext(ui_file)[0] + "_ui.py"
        if not(self.force or newer(ui_file, py_file)):
            return
        try:
            from PyQt4 import uic
            fp = open(py_file, 'w')
            uic.compileUi(ui_file, fp)
            fp.close()
            log.info('compiled %s into %s' % (ui_file, py_file))
        except Exception as e:
            self.warn('Unable to compile user interface %s: %s' % (py_file, e))
            if not os.path.exists(py_file) or not file(py_file).read():
                raise SystemExit(1)
            return

    def compile_rc(self, qrc_file, py_file=None):
        # Search for pyuic4 in python bin dir, then in the $Path.
        if py_file is None:
            py_file = os.path.splitext(qrc_file)[0] + "_rc.py"
        if not(self.force or newer(qrc_file, py_file)):
            return
        import PyQt4
        origpath = os.getenv('PATH')
        path = origpath.split(os.pathsep)
        pyqtfolder = os.path.dirname(PyQt4.__file__)
        path.append(os.path.join(pyqtfolder, 'bin'))
        os.putenv('PATH', os.pathsep.join(path))
        if os.system('pyrcc4 "%s" -o "%s"' % (qrc_file, py_file)) > 0:
            self.warn("Unable to generate python module %s for resource file %s"
                      % (py_file, qrc_file))
            if not exists(py_file) or not file(py_file).read():
                raise SystemExit(1)
        else:
            log.info('compiled %s into %s' % (qrc_file, py_file))
        os.putenv('PATH', origpath)

    def _generate_qrc(self, qrc_file, srcfiles, prefix):
        basedir = os.path.dirname(qrc_file)
        f = open(qrc_file, 'w')
        try:
            f.write('<!DOCTYPE RCC><RCC version="1.0">\n')
            f.write('  <qresource prefix="%s">\n' % cgi.escape(prefix))
            for e in srcfiles:
                relpath = e[len(basedir) + 1:]
                f.write('    <file>%s</file>\n'
                        % cgi.escape(relpath.replace(os.path.sep, '/')))
            f.write('  </qresource>\n')
            f.write('</RCC>\n')
        finally:
            f.close()

    def build_rc(self, py_file, basedir, prefix='/'):
        """Generate compiled resource including any files under basedir"""
        # For details, see http://doc.qt.nokia.com/latest/resources.html
        qrc_file = os.path.join(basedir, '%s.qrc' % os.path.basename(basedir))
        srcfiles = [os.path.join(root, e)
                    for root, _dirs, files in os.walk(basedir) for e in files]
        # NOTE: Here we cannot detect deleted files. In such case, we need
        # to remove .qrc manually.
        if not (self.force or newer_group(srcfiles, py_file)):
            return
        try:
            self._generate_qrc(qrc_file, srcfiles, prefix)
            self.compile_rc(qrc_file, py_file)
        finally:
            os.unlink(qrc_file)

    def run(self):
        basepath = os.path.join(os.path.dirname(__file__), 'shamba')
        for dirpath, _, filenames in os.walk(basepath):
            for filename in filenames:
                if filename.endswith('.ui'):
                    self.compile_ui(os.path.join(dirpath, filename))
                elif filename.endswith('.qrc'):
                    self.compile_rc(os.path.join(dirpath, filename))


class build(_build):
    """
    Overrides build from distutils.command.build 
    in order to also build qt stuff.
    
    """
    def run(self):
        self.run_command("build_qt")
        _build.run(self)

# include the sample_project folder
sif = 'sample_input_files/'
data_files.append(
        ('sample_input_files', 
        [sif+'climate.csv', sif+'input/growth.csv', sif+'input/soilInfo.csv'])
)


cmdclass_ = {
        'build': build,
        'build_qt': build_qt
}


setup(name='shamba',
      version='1.0',
      description='Model and graphical user interface for the SHAMBA project',
      long_description=open('README.md').read(),
      author='Matthieu Hughes',
      author_email = 'matthieu.hughes@ed.ac.uk',
      url='shambatool.wordpress.com',
      packages=[
              'shamba', 'shamba.model', 'shamba.gui', 'shamba.gui.designer', 
              'shamba.rasters', 'shamba.rasters.climate', 
              'shamba.rasters.soil', 'shamba.default_input'],
      package_data={
              'shamba.rasters.climate': ['*.txt'],
              'shamba.rasters.soil': ['hwsd.blw', 'hwsd.hdr', 
                                      'HWSD_data.csv', 'hwsd.bil'],
              'shamba.default_input': ['*.csv']},
      scripts=['shamba/shamba_cl.py', 'shamba.pyw'],
      data_files=data_files,
      cmdclass=cmdclass_,
      **extra 
)
