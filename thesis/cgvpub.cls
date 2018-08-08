%%
%% CGV Style-File fuer CGV Publikationen
%% Autor: Guido Juckeland (juckeland@zhr.tu-dresden.de)
%% angepasst von: Sören König 
%% erweitert für Bachelor/Master von: Ludwig Schmutzler

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cgvpub}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load packages

%Umlaute zulassen
\RequirePackage[latin1]{inputenc}
\RequirePackage[T1]{fontenc}


% Schriftart setzen
\RequirePackage{times}

% Bilder einbinden
\RequirePackage[final]{graphicx}


% URL's zulassen
\RequirePackage{url}



% schoene Listings
\RequirePackage{fancyvrb}

% Mathe-krempel laden
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}

% farbe fuer correctme
\RequirePackage{color}

% enumerates mit benutzerdefinierten labels
\RequirePackage{enumerate}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Provide Options

% Proseminar-Vorlage
\newif\if@proseminar
\@proseminarfalse
\DeclareOption{proseminar}{%
  \@proseminartrue
}

% Hauptseminar-Vorlage
\newif\if@hauptseminar
\@hauptseminarfalse
\DeclareOption{hauptseminar}{%
  \@hauptseminartrue
}


% Grosser Beleg-Vorlage
\newif\if@beleg
\@belegfalse
\DeclareOption{beleg}{%
  \@belegtrue
}

% Diplom Inf-Vorlage
\newif\if@diplominf
\@diplominffalse
\DeclareOption{diplominf}{%
  \@diplominftrue
}

% Diplom IST-Vorlage
\newif\if@diplomist
\@diplomistfalse
\DeclareOption{diplomist}{%
  \@diplomisttrue
}

% Diplom TechMath-Vorlage
\newif\if@diplommedieninf
\@diplommedieninffalse
\DeclareOption{diplommedieninf}{%
  \@diplommedieninftrue
}

% Bachelor Vorlage
\newif\if@bachelorofscience
\@bachelorofsciencefalse
\DeclareOption{bachelorofscience}{%
  \@bachelorofsciencetrue
}

% Master Vorlage
\newif\if@masterofscience
\@masterofsciencefalse
\DeclareOption{masterofscience}{%
  \@masterofsciencetrue
}



% fuer die Frauen das "in" am Titel
\newif\if@female
\@femalefalse
\DeclareOption{female}{%
  \@femaletrue
}

% plain-vorlage article
\newif\if@plainarticle
\@plainarticlefalse
\DeclareOption{plainarticle}{%
  \@plainarticletrue
}

% plain-vorlage article
\newif\if@plainreport
\@plainreportfalse
\DeclareOption{plainreport}{%
  \@plainreporttrue
}

% CGV auf Titelseite 
\newif\if@cgvtitle
\@cgvtitlefalse
\DeclareOption{cgvtitle}{%
  \@cgvtitletrue
}

% References numbered not alpha
\newif\if@bibnum
\@bibnumfalse
\DeclareOption{bibnum}{\@bibnumtrue}

% Abbildungsverzeichnis
\newif\if@lof
\@loffalse
\DeclareOption{lof}{\@loftrue}

% Tabellenverzeichnis
\newif\if@lot
\@lotfalse
\DeclareOption{lot}{\@lottrue}


% Hyperref
\newif\if@hyperref
\@hyperreffalse
\DeclareOption{hyperref}{%
  \@hyperreftrue
}

% Nomenclature
\newif\if@nomencl
\@nomenclfalse
\DeclareOption{nomencl}{%
  \@nomencltrue
}

%german-Option verarbeiten
\newif\if@german
\@germanfalse
\DeclareOption{german}{%
  \@germantrue%
}%

% formatierung fuer cgv-report
\newif\if@final
\@finalfalse
\DeclareOption{final}{%
  \@finaltrue%
}%

% kein Inhaltsverzeichnis
\newif\if@notoc
\@notocfalse
\DeclareOption{notoc}{%
  \@notoctrue%
}%

% keine Aufgabenstellung
\newif\if@noproblem
\@noproblemfalse
\DeclareOption{noproblem}{%
  \@noproblemtrue%
}%

% zweiseitig wenn nicht 'final'
\newif\if@twoside
\@twosidefalse
\DeclareOption{twoside}{%
  \@twosidetrue%
}%
\ProcessOptions


% jetzt ``vergewaltigen'' wir ein paar variablen fuer eigene zwecke
\newcommand\matno[1]{\def\@matno{#1}}
\newcommand\betreuer[1]{\def\@betreuer{#1}}
\newcommand\copyrighterklaerung[1]{\def\@copyrighterklaerung{#1}}
\newcommand\bibfiles[1]{\def\@bibfiles{#1}}
\newcommand\jury[1]{\def\@jury{#1}}
\newcommand\birthday[1]{\def\@birthdate{#1}}
\newcommand\placeofbirth[1]{\def\@birthplace{#1}}
\newcommand\acknowledgments[1]{\def\@acknowledgments{#1}}
\newcommand\problem[1]{\def\@problem{#1}}
%\newif\if@isreport \@isreportfalse \DeclareOption{isreport}{\@isreporttrue}

\def\@acknowledgments{}
\def\@isreport{}
\def\@isplain{}
\def\@isdiploma{}
\def\@bibfiles{}
\def\@copyrighterklaerung{}
\def\@abstractde{}
\def\@abstracten{}
\def\@problem{}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set document type

\if@beleg
  \def\@isreport{1}
\fi
\if@diplominf
  \def\@isreport{1}
  \def\@isdiploma{1}
\fi
\if@diplomist
	\def\@belegtrue{0}
  \def\@isreport{1}
  \def\@isdiploma{1}
\fi
\if@diplommedieninf
  \def\@isreport{1}
  \def\@isdiploma{1}
\fi
\if@plainreport
  \def\@isreport{1}
   \def\@isplain{1}
\fi
\if@plainarticle
  \def\@isplain{1}
\fi
\if@bachelorofscience
  \def\@isreport{1}
  \def\@isdiploma{1}
\fi
\if@masterofscience
  \def\@isreport{1}
  \def\@isdiploma{1}
\fi


\ifx\@empty\@isreport
  \if@final
    \LoadClass[cleardoubleempty, liststotoc, bibtotoc, idxtotoc, 11pt, abstracton, pointlessnumbers, twoside,titlepage,normalheadings]{scrartcl}%
  \else
    \if@twoside
      \LoadClass[cleardoubleempty, liststotoc, bibtotoc, idxtotoc, 11pt, abstracton, pointlessnumbers, twoside,titlepage,normalheadings]{scrartcl}%
    \else
      \LoadClass[cleardoubleempty, liststotoc, bibtotoc, idxtotoc, 11pt, abstracton, pointlessnumbers, oneside,titlepage,normalheadings]{scrartcl}%
    \fi
  \fi
\else
  \if@final
    \LoadClass[cleardoubleempty, liststotoc, bibtotoc, idxtotoc, 11pt, abstracton, pointlessnumbers, twoside,titlepage,openright,normalheadings]{scrreprt}%
  \else
    \if@twoside
      \LoadClass[cleardoubleempty, liststotoc, bibtotoc, idxtotoc, 11pt, abstracton, pointlessnumbers, twoside,titlepage,openright,normalheadings]{scrreprt}%
    \else
      \LoadClass[cleardoubleempty, liststotoc, bibtotoc, idxtotoc, 11pt, abstracton, pointlessnumbers, oneside,titlepage,openright,normalheadings]{scrreprt}%
    \fi
  \fi
\fi

% deckblatttitel, je nachdem, was fuer ein dokument erstellt werden soll

\newcommand{\titleheader}[0]{$\quad$}
\newcommand{\copyrightinfo}[0]{ }
\newcommand{\betreuerlabel}[0]{ }
\newcommand{\matnostart}[0]{ }
\newcommand{\matnolabel}[0]{ }
\newcommand{\matnoclosing}[0]{ }
\newcommand{\birthstart}[0]{ }
\newcommand{\birthmiddle}[0]{ }
\newcommand{\birthclosing}[0]{ }

\if@proseminar%
\renewcommand{\titleheader}[0]{Proseminar ``Computergraphik''}
\fi

\if@hauptseminar%
\renewcommand{\titleheader}[0]{Hauptseminar\\ ``Graphische Datenverarbeitung''}
\fi


\if@beleg%
\renewcommand{\titleheader}[0]{Gro{\ss}er Beleg}
\def\@faculty{Informatik}
\fi

\if@female
 \def\@female{in}
\else
 \def\@female{}
\fi

\if@diplominf%
\renewcommand{\titleheader}[0]{Diplomarbeit \\[1cm]\large zur Erlangung des akademischen Grades\\ Diplom-Informatiker\@female}
\def\@faculty{Informatik}
\fi

\if@diplomist%
\renewcommand{\titleheader}[0]{Diplomarbeit \\[1cm]\large zur Erlangung des akademischen Grades\\ Diplomingenieur\@female~f\"{u}r Informationssystemtechnik}
\def\@faculty{Elektrotechnik und Informationstechnik}
\fi

\if@diplommedieninf%
\renewcommand{\titleheader}[0]{Diplomarbeit \\[1cm]\large zur Erlangung des akademischen Grades\\ Diplom-Medieninformatiker\@female}
\def\@faculty{Informatik}
\fi

\if@bachelorofscience%
\renewcommand{\titleheader}[0]{Bachelorarbeit \\[1cm]\large zur Erlangung des akademischen Grades\\ Bachelor of Science}
\def\@faculty{Informatik}
\fi

% \if@masterofscience%
% \renewcommand{\titleheader}[0]{Masterarbeit \\[1cm]\large zur Erlangung des akademischen Grades\\ Master of Science}
% \def\@faculty{Informatik}
% \fi

\if@masterofscience%
\renewcommand{\titleheader}[0]{Master Thesis \\[1cm]\large Master Computational Logic}
\def\@faculty{Informatik}
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set page-layout

% Definition der Seitenraender
%\setlength{\hoffset}{-1in}
%\addtolength{\hoffset}{2cm}
%\setlength{\voffset}{-1in}
%\addtolength{\voffset}{2cm}
%\addtolength{\voffset}{-\baselineskip}
%\setlength{\topmargin}{0mm}
%\setlength{\headheight}{2.0\baselineskip}
%\setlength{\headsep}{\baselineskip}
\setlength{\textheight}{24.5cm}
\setlength{\textwidth}{16cm}
\setlength{\evensidemargin}{0mm}
\setlength{\oddsidemargin}{0mm}
%\setlength{\marginparsep}{0mm}
%\setlength{\marginparwidth}{0mm}
%\setlength{\marginparpush}{0mm}
%\setlength{\footskip}{\baselineskip}
\setlength{\parindent}{0mm}
\setlength{\parskip}{2mm}
\setlength{\partopsep}{0mm}
\renewcommand{\cleardoublepage}{\cleardoublestandardpage}
\def\@chapapp{}
% enhanced tables
\RequirePackage{tabularx}
\RequirePackage{longtable}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Option "german" bearbeiten

%Neudeutsch laden
\if@german%
\RequirePackage{ngerman}%
\fi

%Da man bei Neudeutsch ebenfalls Zitate in Altdeutsch haben wird,
%Wird das german-Paket auf jeden Fall benoetigt.
\if@german%
  \RequirePackage{german}%
  \def\@copyrightlabel{Erkl\"{a}rungen zum Urheberrecht}
  \renewcommand\copyrightinfo[0]{%
    \cleardoublepage
    \ifx\@empty\@isreport
      \section*{\@copyrightlabel}
    \else
      \chapter*{\@copyrightlabel}
    \fi
    \@copyrighterklaerung}
  \ifx\@empty\@isplain
    \renewcommand{\betreuerlabel}[0]{Betreuer: }
		\renewcommand{\matnostart}[0]{(}
    \renewcommand{\matnolabel}[0]{Mat.-Nr.: }
    \renewcommand{\matnoclosing}[0]{)}
    \renewcommand{\birthstart}[0]{(Geboren am }
    \renewcommand{\birthmiddle}[0]{ in }
    \renewcommand{\birthclosing}[0]{, }
  \fi
\else
  \def\@copyrightlabel{Copyright Information}
  \renewcommand\copyrightinfo[0]{%
    \cleardoublepage
    \ifx\@empty\@isreport
    \section*{\@copyrightlabel}
  \else
    \chapter*{\@copyrightlabel}
  \fi
  \@copyrighterklaerung}
  \ifx\@empty\@isplain
    \renewcommand{\betreuerlabel}[0]{Supervisor: }
		\renewcommand{\matnostart}[0]{(}
    \renewcommand{\matnolabel}[0]{Mat.-No.: }
    \renewcommand{\matnoclosing}[0]{)}
    \renewcommand{\birthstart}[0]{(Born }
    \renewcommand{\birthmiddle}[0]{ in }
    \renewcommand{\birthclosing}[0]{, }
  \fi
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Hyperref laden, wenn gewuenscht
\if@hyperref
\RequirePackage[pdftitle=title,pdfauthor=author,pdfstartview={FitH}, hyperindex=true, bookmarks=true, bookmarksnumbered=true, plainpages=false, pdfborder={0 0 0}]{hyperref}
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Nomenclature laden
\if@nomencl
  \if@german
    \RequirePackage[german]{nomencl}
  \else
    \RequirePackage{nomencl}
  \fi

  % Randbreite definieren
  \def\@nomenclmargin{3cm}
  \newcommand\setnomenclmargin[1]{\def\@nomenclmargin{#1}}
  
  % index erstellen
  \makeglossary 

\fi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Abstracts zusammenbauen
\newcommand{\abstractde}[1]{\def\@abstractde{#1}}
\newcommand{\abstracten}[1]{\def\@abstracten{#1}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Gliederung und Numerierung
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% kopfzeile auch bei chapters
\ifx\@empty\@isreport
\else
\renewcommand*{\chapterpagestyle}{fancy}
\fi

% kopfzeile auch bei abstract
\let\@altab=\abstract
\let\end@altab=\endabstract
\renewenvironment{abstract}{%
\fancyfoot{}
\fancyhead{}
\if@final
  \fancyhead[LE,RO]{\thepage}
\else
  \if@twoside
    \fancyhead[LE,RO]{\thepage}
  \else
    \fancyhead[R]{\thepage}
  \fi
\fi
\begin{@altab}%
\thispagestyle{fancy}}{\end{@altab}%
\cleardoublepage
\ifx\@empty\@isreport
  \if@final
    \fancyhead[LO]{\@author}
    \fancyhead[RE]{\@title}
  \else
    \if@twoside
      \fancyhead[LO]{\@author}
      \fancyhead[RE]{\@title}
    \else
      \fancyhead[L]{\@author -- \@title}
    \fi
  \fi
\else
  \if@final
    \fancyhead[LO]{\rightmark}
    \fancyhead[RE]{\leftmark}
  \else
    \if@twoside
      \fancyhead[LO]{\rightmark}
      \fancyhead[RE]{\leftmark}
    \else
      \fancyhead[L]{\rightmark}
    \fi
  \fi
\fi}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Institution fuer Titelseite
\if@cgvtitle
  \if@german
  \newcommand{\@institution}{
    \begin{center}
    %Institution an der die Arbeit geschrieben wurde
    \huge
    \textsc{Technische Universit"at Dresden}

    \bigskip
    \Large
    \textsc{Institut f"ur Software- und Multimediatechnik\\
    Prof. Dr. Stefan Gumhold\\
    $\quad$\\
    $\quad$}
    \end{center}
  }
  \else
  \newcommand{\@institution}{
    \begin{center}
    %Institution an der die Arbeit geschrieben wurde
    \huge
    \textsc{Technical University Dresden}

    \bigskip
    \Large
    \textsc{Institute of Software- and Multimedia Technology\\
    Prof. Dr. Stefan Gumhold\\
    $\quad$\\
    $\quad$}
    \end{center}
  }
  \fi
  \else
	  \if@german
	  \newcommand{\@institution}{
	    \begin{center}
	    %Institution an der die Arbeit geschrieben wurde
	    \huge
	    \textsc{Technische Universit"at Dresden}

	    \bigskip
	    \Large
	    \textsc{Fakult"at Informatik\\
	    Institut f"ur Software- und Multimediatechnik\\
	    Professur f"ur Computergraphik und Visualisierung\\
	    Prof. Dr. Stefan Gumhold}
	    \end{center}
	  }
	  \else
	  \newcommand{\@institution}{
	    \begin{center}
	    %Institution an der die Arbeit geschrieben wurde
	    \huge
	    \textsc{Technische Universität Dresden}

	    \bigskip
	    \Large
     \textsc{Faculty of Computer Science\\
	    International Center For Computational Logic}
	    \end{center}
	  }
	  \fi
  \fi
%\fi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Style fuer Bibliography
\if@bibnum
  \if@german
    \bibliographystyle{plaindin}
  \else
    \bibliographystyle{plain}
  \fi
\else
  \if@german
    \bibliographystyle{alphadin}
  \else
    \bibliographystyle{alpha}
  \fi
\fi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% List of tables/figures hinzu falls gewuenscht
\if@lof
  \newcommand{\printlof}{\listoffigures}
\else
  \newcommand{\printlof}{ }
\fi
\if@lot
  \newcommand{\printlot}{\listoftables}
\else
  \newcommand{\printlot}{ }
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% zeile unter dem autor zusammenstellen
\ifx\@empty\@isdiploma
  \ifx\@empty\@isplain
    \def\@subauthor{\matnostart\matnolabel\@matno\matnoclosing}
  \else
    \def\@subauthor{$\quad$}
  \fi
\else
  \def\@subauthor{\birthstart\@birthdate\birthmiddle\@birthplace\birthclosing\matnolabel\@matno\matnoclosing}
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Literatur-, Abbildungs- und Tabellenverzeichnis vor dem Anhang
\def\@haveapp{}
\let\@altapp=\appendix
\renewcommand{\appendix}[0]{\cleardoublestandardpage
\ifx\@empty\@bibfiles
\else
  \bibliography{\@bibfiles}
\fi
\cleardoublestandardpage
\printlof
\cleardoublestandardpage
\printlot
\cleardoublestandardpage
\@altapp
\def\@haveapp{1}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Kopfzeile
\RequirePackage{fancyhdr}

% Automatische Generierung des Dokumentgrundgeruests
\let\@altdoc=\document
\let\end@altdoc=\enddocument
\renewenvironment{document}{%
\begin{@altdoc}%
%\thispagestyle{empty}
%\pagestyle{empty}
\ifx\@empty\@isplain
\else
  \def\@publishers{$\quad$}
  \def\@subject{$\quad$}
\fi
\begin{titlepage}
  \enlargethispage{1cm}

  \@institution
  \vspace{2cm}

  \begin{center}
  % Titel der Arbeit
  \huge
  \titleheader\\[1.5cm]
  \huge
  \@title
  \vspace{2cm}

  % Zweck der Arbeit
  \Large
  {\Large \@author }\\\@subauthor\\[2cm]

  \end{center}
  \vfill

\begin{flushleft}
 \Large
\ifx\@empty\@isplain
 \betreuerlabel \@betreuer\\[1cm]
\else
 $\quad$\\[1cm]
\fi
        Dresden, \@date\\[2cm]
\end{flushleft}
\end{titlepage}
   \cleardoubleemptypage

% zeilenabstand ja nach variante
\if@final
  \renewcommand{\baselinestretch}{1.0}
\else
  \renewcommand{\baselinestretch}{1.5}
\fi
\normalsize
% Seitenstil festlegen
\fancyfoot{}
\fancyhead{}
\pagestyle{fancy}
\thispagestyle{fancy}
% Aufgabenstellung
\ifx\@isbeleg\@isdiploma
\else
  \if@noproblem
  \else
  	\ifx\@haupseminartrue\@proseminar
  		\section*{Aufgabenstellung}
  	\else
    	\chapter*{Aufgabenstellung}
    \fi
  	\@problem
  \fi
\fi

\cleardoublestandardpage
%Selbständigkeitserklärung
     \chapter*{Selbstst\"{a}ndigkeitserkl\"{a}rung}

    Hiermit erkl\"{a}re ich, dass ich die von mir am heutigen Tag dem Pr\"{u}fungsausschuss der Fakult\"{a}t \@faculty~eingereichte Arbeit zum Thema:
     \begin{center}
     \textit{\@title} 
     \end{center}
     
     vollkommen selbstst\"{a}ndig verfasst und keine anderen als die angegebenen Quellen und Hilfsmittel benutzt sowie Zitate kenntlich gemacht habe.

    Dresden, den \@date\\[3cm]
    \@author

\cleardoublestandardpage
%Zusammenfassung
\ifx\@empty\@isdiploma
\else
  \renewcommand{\baselinestretch}{1.0}
  {
  \chapter*{Abstract}
  \@abstracten
  }
  \if@final
  \renewcommand{\baselinestretch}{1.0}
\else
  \renewcommand{\baselinestretch}{1.5}
\fi   
\fi
\cleardoublestandardpage
%Inhaltsverzeichnis
\pagenumbering{arabic}
\if@final
  \fancyhead[LE,RO]{\thepage}
\else
  \if@twoside
    \fancyhead[LE,RO]{\thepage}
  \else
    \fancyhead[R]{\thepage}
  \fi
\fi
\if@notoc
\else
  \tableofcontents
  \cleardoublestandardpage
\fi
%Nomenclature
\if@nomencl
  \if@german
    \addcontentsline{toc}{chapter}{Symbolverzeichnis}
  \else
    \addcontentsline{toc}{chapter}{Nomenclature}
  \fi
  \label{sec:Nomenclature}
  \printglossary[\@nomenclmargin]
\fi
%Kopfzeile basteln
\markboth{}{}
\ifx\@empty\@isreport
  \if@final
    \fancyhead[LO]{\@author}
    \fancyhead[RE]{\@title}
  \else
    \if@twoside
      \fancyhead[LO]{\@author}
      \fancyhead[RE]{\@title}
    \else
      \fancyhead[L]{\@author -- \@title}
    \fi
  \fi
\else
  \if@final
    \fancyhead[LO]{\rightmark}
    \fancyhead[RE]{\leftmark}
  \else
    \if@twoside
      \fancyhead[LO]{\rightmark}
      \fancyhead[RE]{\leftmark}
    \else   
      \fancyhead[L]{\leftmark}
    \fi
  \fi
\fi}{%
% Literatur-, Abbildungs- und Tabellenverzeichnis, wenn nicht schon bei
% \appendix automatisch davor eingefügt
\ifx\@empty\@haveapp
  \cleardoublestandardpage
  \ifx\@empty\@bibfiles
  \else
    \bibliography{\@bibfiles}
  \fi
  \cleardoublestandardpage
  \printlof
  \cleardoublestandardpage
  \printlot
\fi
\cleardoublestandardpage
%Danksagung
\ifx\@empty\@acknowledgments
\else
  \if@german
    \@mkboth{Danksagung}{}
  \else
    \@mkboth{Acknowledgments}
  \fi
  \ifx\@empty\@isreport
    \if@german
      \section*{Danksagung}
    \else
      \section*{Acknowledgments}
    \fi
  \else
    \if@german
      \chapter*{Danksagung}
    \else
      \chapter*{Acknowledgments}
    \fi
  \fi
  \@acknowledgments
\fi
% Copyright-Kram
\cleardoublestandardpage
\ifx\@empty\@copyrighterklaerung
\else
  \@mkboth{\@copyrightlabel}{}
  \copyrightinfo
\fi
\cleardoublestandardpage
\end{@altdoc}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Nuetzliche Zusatzkommandos

\newcommand\longcite[2]{(vgl.~\cite{#1}, S.~#2)}

\newcommand\shortcite[2]{(\cite{#1}, S.~#2)}

% itemize ohne abstand zw. den items
\let\@altitemize=\itemize
\let\end@altitemize=\enditemize
\renewenvironment{itemize}{%
\begin{@altitemize}%
\ifnum \@itemdepth = 1
  \vspace{-10pt}
  \setlength{\itemsep}{-1pt}
\else
  \ifnum \@itemdepth = 2
    \vspace{-6pt}
    \setlength{\itemsep}{0pt}
  \else
    \ifnum \@itemdepth = 3
      \vspace{-2pt}
      \setlength{\itemsep}{1pt}
    \else
      \vspace{0pt}
      \setlength{\itemsep}{2pt}
    \fi
  \fi
\fi
  \setlength{\parsep}{0pt}
\setlength{\labelsep}{12pt}
}%
{\end{@altitemize}\vspace{-2pt}}

% enumerate ohne abstand zw. den items
\def\enumerate{%
  \ifnum \@enumdepth >3 \@toodeep\else
	\ifnum \@enumdepth = 0
	  \vspace{-10pt}
	  \setlength{\itemsep}{-1pt}
	\else
	  \ifnum \@enumdepth = 1
	    \vspace{-6pt}
	    \setlength{\itemsep}{0pt}
	  \else
	    \ifnum \@enumdepth = 2
	      \vspace{-2pt}
	      \setlength{\itemsep}{1pt}
	    \else
	      \vspace{0pt}
	      \setlength{\itemsep}{2pt}
	    \fi
	  \fi
	\fi
	\setlength{\parsep}{0pt}
	\setlength{\labelsep}{12pt}
      \advance\@enumdepth \@ne
      \edef\@enumctr{enum\romannumeral\the\@enumdepth}\fi
  \@ifnextchar[{\@@enum@}{\@enum@}}
\def\@@enum@[#1]{%
  \@enLab{}\let\@enThe\@enQmark
  \@enloop#1\@enum@
  \ifx\@enThe\@enQmark\@warning{The counter will not be printed.%
   ^^J\space\@spaces\@spaces\@spaces The label is: \the\@enLab}\fi
  \expandafter\edef\csname label\@enumctr\endcsname{\the\@enLab}%
  \expandafter\let\csname the\@enumctr\endcsname\@enThe
  \csname c@\@enumctr\endcsname7
  \expandafter\settowidth
            \csname leftmargin\romannumeral\@enumdepth\endcsname
            {\the\@enLab\hspace{\labelsep}}%
  \@enum@}
\def\@enum@{\list{\csname label\@enumctr\endcsname}%
           {\usecounter{\@enumctr}\def\makelabel##1{\hss\llap{##1}}}}

% paragraph mit Zeilenumbruch nach der Ueberschrift
\newcommand\Paragraph[1]{\minisec{#1}}

% im text korrekturen anzeigen
\newcommand{\correctme}[1]{\textcolor{red}{#1}}

% korrekturen ueber mehrere absaetze
\newenvironment{correctmore}{\color{red}}{\color{black}}


%listings
\RequirePackage{listings}


\definecolor{KeywordBlue}{cmyk}{0.88,0.77,0,0} %88,77,0,0
\definecolor{CommentGreen}{cmyk}{0.87,0.24,1.0,0.13} %87,24,100,13
\lstset{basicstyle=\scriptsize\ttfamily,language=C,commentstyle=\color{CommentGreen},keywordstyle=\ttfamily\color{KeywordBlue} }