
# Version 3.5 09/10/25
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean
# Updated July 2026 laelume aka Ashlae Blum(e)

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2025

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Logging for batch processing

import os
import time
import os
import time
import logging

VERBOSE = True    # toggle verbose debug logging on/off for this module

logger = logging.getLogger("batch_log")
logger.setLevel(logging.DEBUG if VERBOSE else logging.WARNING)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(_handler)

class Log(object):
    """ Used for logging info during batch processing.
        Stores most recent analysis for each species, to stay in sync w/ data files.
        Arguments:
        1. path to log file
        2. species
        3. list of other settings of the current analysis

        LOG FORMAT, for each analysis:
        #freetext line
        species
        settings line
        files, multiple lines
    """

    def __init__(self, path, species, settings):
        # in order to append, the previous log must:
        # 1. exist
        # 2. be writeable
        # 3. match current analysis
        # On init, we parse the existing log to see if appending is possible.
        # Actual append/create happens later.
        self.possibleAppend = False
        self.filepath = path
        # self.file will be an IO stram opened by the main launcher
        self.species = species
        # Convert settings dict to formatted string
        if isinstance(settings, dict):
            parts = []
            for key, value in settings.items():
                if value and value != "None" and value != False:
                    parts.append(f"{key}={value}")
            self.settings = ', '.join(parts) if parts else 'default'
        else:
            # Legacy support for list format
            self.settings = ','.join(map(str, settings))
        self.oldAnalyses = []
        self.filesDone = []
        self.currentHeader = ""
        allans = []

        # now, check if the specified log can be resumed:
        if os.path.isfile(path):
            try:
                f = open(path, 'r+')
                print("Found log file at %s" % path)

                lines = [line.rstrip('\n') for line in f]
                f.close()
                lstart = 0
                lend = 1
                # parse to separate each analysis into
                # [freetext, species, settings, [files]]
                # (basically I'm parsing txt into json because I'm dumb)
                while lend<len(lines):
                    #print(lines[lend])
                    if len(lines[lend]) > 0:
                        if lines[lend][0] == "#":
                            allans.append([lines[lstart], lines[lstart+1], lines[lstart+2],
                                            lines[lstart+3 : lend]])
                            lstart = lend
                    lend += 1
                allans.append([lines[lstart], lines[lstart+1], lines[lstart+2],
                                lines[lstart+3 : lend]])

                # parse the log thusly:
                # if current species analysis found, store parameters
                # and compare to check if it can be resumed.
                # store all other analyses for re-printing.
                for a in allans:
                    #print(a)
                    if a[1]==self.species:
                        print("Resumable analysis found")
                        # do not reprint this in log
                        if a[2]==self.settings:
                            self.currentHeader = a[0]
                            # (a1 and a2 match species & settings anyway)
                            self.filesDone = a[3]
                            self.possibleAppend = True
                    else:
                        # store this for re-printing to log
                        self.oldAnalyses.append(a)

            except IOError:
                # bad error: lacking permissions?
                print("ERROR: could not open log at %s" % path)

    # def appendFile(self, filename):
    #     print('Appending %s to log' % filename)
    #     # convert to path relative to the log file directory
    #     if os.path.isabs(filename):
    #         filename = os.path.relpath(filename, os.path.dirname(self.filepath))

    #     # attach file path to end of log
    #     self.file.write(filename)
    #     self.file.write("\n")
    #     self.file.flush()

    def appendFile(self, filename, annotationsFound=None):
        """ Appends a processed file to the log, with a timestamp and whether any annotations were found.

        filename:         path to the processed file, converted to a path relative
                           to the log file's directory before writing
        annotationsFound: True/False if known, None to omit the field entirely
                           (keeps old-format-compatible lines when the caller
                           doesn't have this information)
        """
        print('Appending %s to log' % filename)
        # convert to path relative to the log file directory
        if os.path.isabs(filename):
            filename = os.path.relpath(filename, os.path.dirname(self.filepath))

        # tags annotation with date and time processed
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if annotationsFound is None:
            line = f"{timestamp} | {filename}"
        else:
            line = f"{timestamp} | annotations={annotationsFound} | {filename}"

        # attach file path to end of log
        self.file.write(line)
        self.file.write("\n")
        self.file.flush()

    def appendDirectoryComplete(self, dirpath):
        """ Appends a marker line noting that every file in a given directory has been processed.

        dirpath: path to the completed directory, converted to a path relative
                 to the log file's directory before writing, matching the same
                 convention used by appendFile for individual files

        Uses a ">>>" prefix rather than "#", since the log parser in __init__
        identifies the start of a new analysis block by checking for a leading
        "#" character; a marker line starting with "#" would be misread as a
        new analysis header and corrupt the block structure.
        """
        if os.path.isabs(dirpath):
            dirpath = os.path.relpath(dirpath, os.path.dirname(self.filepath))

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f">>> DIR COMPLETE: {dirpath} | {timestamp}"

        self.file.write(line)
        self.file.write("\n")
        self.file.flush()
        logger.debug("Directory marked complete: %s", dirpath)


    # def getDoneFiles(self, possiblefiles):
    #     """ Selects files that are stored in this log from possiblefiles.
    #         Assumes possiblefiles stores absolute paths. """
    #     currdir = os.path.dirname(self.filepath)
    #     done_abs = [os.path.normpath(os.path.join(currdir, f)) for f in self.filesDone if not os.path.isabs(f)]
    #     # assuming relative paths on both lists:
    #     out = set(done_abs).intersection(set(possiblefiles))
    #     return(out)



    def getDoneFiles(self, possiblefiles):
        """ Selects files that are stored in this log from possiblefiles.
            Assumes possiblefiles stores absolute paths.

        Parses both current-format lines (TIMESTAMP | annotations=bool | path,
        or TIMESTAMP | path) and older bare-path lines from logs written before
        this format existed. Directory-completion marker lines (>>> DIR COMPLETE...)
        are skipped, since they are not individual file entries.
        """
        currdir = os.path.dirname(self.filepath)

        parsedPaths = []
        for f in self.filesDone:
            if f.startswith(">>> DIR COMPLETE"):
                continue
            # current format has one or two " | " separators; the path is
            # always the last segment
            path = f.rsplit(" | ", 1)[-1]
            parsedPaths.append(path)

        done_abs = [os.path.normpath(os.path.join(currdir, f)) for f in parsedPaths if not os.path.isabs(f)]
        # assuming relative paths on both lists:
        out = set(done_abs).intersection(set(possiblefiles))
        return(out)


    def appendHeader(self, header, species, settings):
        if header is None:
            header = "#Analysis started on " + time.strftime("%Y %m %d, %H:%M:%S") + ":"
        self.file.write(header)
        self.file.write("\n")
        self.file.write(species)
        self.file.write("\n")
        if type(settings) is list:
            settings = ','.join(settings)
        self.file.write(settings)
        self.file.write("\n")
        self.file.flush()

    def reprintOld(self):
        # push everything from oldAnalyses to log
        # To be called once starting a new log is confirmed
        for a in self.oldAnalyses:
            self.appendHeader(a[0], a[1], a[2])
            for f in a[3]:
                self.appendFile(f)
