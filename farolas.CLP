;; =========================================================
;; ===     A N C H U R A   Y    P R O F U N D I D A D   ====
;; =========================================================

(defglobal ?*nod-gen* = 0)

(deffacts mapa
    (Dimension 5 5)
    (Almacen 2 3)
    (Robot 1 1 0 Farolas Farola 3 5 2 Farola 4 3 3 Farola 5 5 2 nivel 0)
    (MaxBombillas 3)
    (CapRobot 1 2 3)
)

(defrule coger
    ?g <- (Robot ?c ?f ?b $?p nivel ?nivel)
    (Almacen ?columna ?filita)
    (MaxBombillas ?max)
    (CapRobot $?a ?w $?z)
    (profundidad-maxima ?prof)
    (test (and (= ?c ?columna)(= ?f ?filita)))
    (test (or (<(+ ?w ?b) ?max) (= (+ ?w ?b) ?max)))
    ;(test (or (<(+ ?w ?b) ?total) (=(+ ?w ?b) ?total)))
    (test (< ?nivel ?prof))
    =>
    (assert (Robot ?c ?f (+ ?b ?w) $?p nivel (+ ?nivel 1)))
    (bind ?*nod-gen* (+ ?*nod-gen* 1))
)

(defrule cambiar
    ?g<-(Robot ?c ?f ?b Farolas $?x Farola ?cf ?ff ?bf $?y nivel ?nivel)
    (profundidad-maxima ?prof)
    (test (or (and (or (= (+ ?c 1) ?cf)(= (- ?c 1) ?cf))(= ?f ?ff))(and (or (= (+ ?f 1) ?ff)(= (- ?f 1) ?ff))(= ?c ?cf))))
    (test (or (> ?b ?bf) (= ?b ?bf)))
    (test (< ?nivel ?prof))
    =>
    (assert (Robot ?c ?f (- ?b ?bf) Farolas $?x $?y nivel (+ ?nivel 1)))
    (bind ?*nod-gen* (+ ?*nod-gen* 1))
)

(defrule derecha
    ?g<-(Robot ?c ?f ?b Farolas $?LF nivel ?nivel)
    (Dimension ?dimX ?dimY)
    (profundidad-maxima ?prof)
    (test (< ?c ?dimX))
    (test (not (member$ (create$ Farola (+ ?c 1) ?f) $?LF)))
    (test (< ?nivel ?prof))
    =>
    (assert (Robot (+ ?c 1) ?f ?b Farolas $?LF nivel (+ ?nivel 1)))
    (bind ?*nod-gen* (+ ?*nod-gen* 1))
)

(defrule izquierda
    ?g<-(Robot ?c ?f ?b Farolas $?LF nivel ?nivel)
    (Dimension ?dimX ?dimY)
    (profundidad-maxima ?prof)
    (test (> ?c 1))
    (test (not (member$ (create$ Farola (- ?c 1) ?f) $?LF)))
    (test (< ?nivel ?prof))
    =>
    (assert (Robot (- ?c 1) ?f ?b Farolas $?LF nivel (+ ?nivel 1)))
    (bind ?*nod-gen* (+ ?*nod-gen* 1))
)

(defrule arriba
    ?g<-(Robot ?c ?f ?b Farolas $?LF nivel ?nivel)
    (Dimension ?dimX ?dimY)
    (profundidad-maxima ?prof)
    (test (< ?f ?dimY))
    (test (not (member$ (create$ Farola ?c (+ ?f 1)) $?LF)))
    (test (< ?nivel ?prof))
    =>
    (assert (Robot ?c (+ ?f 1) ?b Farolas $?LF nivel (+ ?nivel 1)))
    (bind ?*nod-gen* (+ ?*nod-gen* 1))
)

(defrule abajo
    ?g<-(Robot ?c ?f ?b Farolas $?LF nivel ?nivel)
    (Dimension ?dimX ?dimY)
    (profundidad-maxima ?prof)
    (test (> ?f 1))
    (test (not (member$ (create$ Farola ?c (- ?f 1)) $?LF)))
    (test (< ?nivel ?prof))
    =>
    (assert (Robot ?c (- ?f 1) ?b Farolas $?LF nivel (+ ?nivel 1)))
    (bind ?*nod-gen* (+ ?*nod-gen* 1))
)

;; ========================================================
;; =========      ESTRATEGIA DE CONTROL DE BUSQUEDA    ========
;; ========================================================
;; La regla 'objetivo' se utiliza para detectar cuando se ha alcanzado el estado objetivo

(defrule objetivo
    (declare (salience 100))
    ?g<-(Robot ?x ?y 0 Farolas nivel ?n )

    =>
    (printout t "SOLUCION ENCONTRADA EN EL NIVEL " ?n crlf)
    (printout t "NUMERO DE NODOS EXPANDIDOS O REGLAS DISPARADAS " ?*nod-gen* crlf)
    (printout t "HECHO OBJETIVO " ?g crlf)

    (halt)
)

(defrule no_solucion
    (declare (salience -99))
    (Robot $? nivel ?nivel $?)

    =>
    (printout t "SOLUCION NO ENCONTRADA" crlf)
    (printout t "NUMERO DE NODOS EXPANDIDOS O REGLAS DISPARADAS " ?*nod-gen* crlf)

    (halt)
)


(deffunction inicio ()
        (reset)
	(printout t "Profundidad Maxima:= " )
	(bind ?prof (read))
	(printout t "Tipo de Busqueda " crlf "    1.- Anchura" crlf "    2.- Profundidad" crlf )
	(bind ?a (read))
	(if (= ?a 1)
	       then    (set-strategy breadth)
	       else   (set-strategy depth))
        (printout t " Ejecuta run para poner en marcha el programa " crlf)
	(assert (profundidad-maxima ?prof))
)

(deffunction camino
	(?g)
	(bind ?lista (fact-slot-value ?g implied))
	(bind ?l2 (member$ nivel ?lista))
	(bind ?n (nth (+ ?l2 1) ?lista))
	;;(printout t "Nivel=" ?n crlf)
	(bind ?dir (nth (length ?lista) ?lista))
	(bind ?mov (subseq$ ?lista (+ ?l2 3) (- (length ?lista) 2)))
	(bind ?path (create$ ?dir ?mov))
	;;(printout t ?dir "    " ?mov crlf)

	(loop-for-count (- ?n 1)
		(bind ?lista (fact-slot-value (fact-index ?dir) implied))
		(bind ?dir (nth (length ?lista) ?lista))
		(bind ?l2 (member$ nivel ?lista))
		(bind ?mov (subseq$ ?lista (+ ?l2 3) (- (length ?lista) 2)))
		(bind ?path (create$ ?dir ?mov ?path))
	)

	(printout t "Camino: " ?path crlf)
)