(set-info :original "/tmp/sea-Qvs1my/04.pp.ms.o.bc")
(set-info :authors "SeaHorn v.0.1.0-rc3")
(declare-rel verifier.error (Bool Bool Bool ))
(declare-rel main@entry ())
(declare-rel main@_bb (Int Int ))
(declare-rel main@verifier.error.split ())
(declare-var main@%_5_0 Bool )
(declare-var main@%_4_0 Bool )
(declare-var main@%y.0.i2.lcssa_1 Int )
(declare-var main@%x.0.i1_2 Int )
(declare-var main@entry_0 Bool )
(declare-var main@%_0_0 Int )
(declare-var main@_bb_0 Bool )
(declare-var main@%y.0.i2_0 Int )
(declare-var main@%x.0.i1_0 Int )
(declare-var main@%y.0.i2_1 Int )
(declare-var main@%x.0.i1_1 Int )
(declare-var main@%_2_0 Int )
(declare-var main@%_3_0 Int )
(declare-var main@_bb_1 Bool )
(declare-var main@%y.0.i2_2 Int )
(declare-var main@verifier.error_0 Bool )
(declare-var main@%y.0.i2.lcssa_0 Int )
(declare-var main@verifier.error.split_0 Bool )
(rule (verifier.error false false false))
(rule (verifier.error false true true))
(rule (verifier.error true false true))
(rule (verifier.error true true true))
(rule main@entry)
(rule (=> (and main@entry
         true
         (=> main@_bb_0 (and main@_bb_0 main@entry_0))
         main@_bb_0
         (=> (and main@_bb_0 main@entry_0) (= main@%y.0.i2_0 main@%_0_0))
         (=> (and main@_bb_0 main@entry_0) (= main@%x.0.i1_0 (- 50)))
         (=> (and main@_bb_0 main@entry_0) (= main@%y.0.i2_1 main@%y.0.i2_0))
         (=> (and main@_bb_0 main@entry_0) (= main@%x.0.i1_1 main@%x.0.i1_0)))
    (main@_bb main@%y.0.i2_1 main@%x.0.i1_1)))
(rule (=> (and (main@_bb main@%y.0.i2_0 main@%x.0.i1_0)
         true
         (= main@%_2_0 (+ main@%y.0.i2_0 main@%x.0.i1_0))
         (= main@%_3_0 (+ main@%y.0.i2_0 1))
         (= main@%_4_0 (< main@%_2_0 0))
         (=> main@_bb_1 (and main@_bb_1 main@_bb_0))
         main@_bb_1
         (=> (and main@_bb_1 main@_bb_0) main@%_4_0)
         (=> (and main@_bb_1 main@_bb_0) (= main@%y.0.i2_1 main@%_3_0))
         (=> (and main@_bb_1 main@_bb_0) (= main@%x.0.i1_1 main@%_2_0))
         (=> (and main@_bb_1 main@_bb_0) (= main@%y.0.i2_2 main@%y.0.i2_1))
         (=> (and main@_bb_1 main@_bb_0) (= main@%x.0.i1_2 main@%x.0.i1_1)))
    (main@_bb main@%y.0.i2_2 main@%x.0.i1_2)))
(rule (let ((a!1 (and (main@_bb main@%y.0.i2_0 main@%x.0.i1_0)
                true
                (= main@%_2_0 (+ main@%y.0.i2_0 main@%x.0.i1_0))
                (= main@%_3_0 (+ main@%y.0.i2_0 1))
                (= main@%_4_0 (< main@%_2_0 0))
                (=> main@verifier.error_0
                    (and main@verifier.error_0 main@_bb_0))
                (=> (and main@verifier.error_0 main@_bb_0) (not main@%_4_0))
                (=> (and main@verifier.error_0 main@_bb_0)
                    (= main@%y.0.i2.lcssa_0 main@%y.0.i2_0))
                (=> (and main@verifier.error_0 main@_bb_0)
                    (= main@%y.0.i2.lcssa_1 main@%y.0.i2.lcssa_0))
                (=> main@verifier.error_0
                    (= main@%_5_0 (> main@%y.0.i2.lcssa_1 (- 1))))
                (=> main@verifier.error_0 (not main@%_5_0))
                (=> main@verifier.error.split_0
                    (and main@verifier.error.split_0 main@verifier.error_0))
                main@verifier.error.split_0)))
  (=> a!1 main@verifier.error.split)))
(query main@verifier.error.split)

