(set-info :original "/tmp/sea-zKcwOL/09.pp.ms.o.bc")
(set-info :authors "SeaHorn v.0.1.0-rc3")
(declare-rel verifier.error (Bool Bool Bool ))
(declare-rel main@entry (Int ))
(declare-rel main@_bb (Int ))
(declare-rel main@.lr.ph7 (Int Int Int ))
(declare-rel main@_bb1 (Int Int Int ))
(declare-rel main@.lr.ph (Int Int Int ))
(declare-rel main@verifier.error.split ())
(declare-var main@%_19_0 Bool )
(declare-var main@%_20_0 Bool )
(declare-var main@%j.0.i4_2 Int )
(declare-var main@%k.1.i3_2 Int )
(declare-var main@%_16_0 Bool )
(declare-var main@%_13_0 Int )
(declare-var main@%_14_0 Int )
(declare-var main@%_15_0 Bool )
(declare-var main@%_9_0 Int )
(declare-var main@%_10_0 Int )
(declare-var main@%_11_0 Bool )
(declare-var main@%.lcssa13_1 Int )
(declare-var main@%.lcssa_1 Int )
(declare-var main@%i.1.i6_2 Int )
(declare-var main@%k.0.i5_2 Int )
(declare-var main@%_4_0 Int )
(declare-var main@%_5_0 Int )
(declare-var main@%_6_0 Bool )
(declare-var main@%_1_0 Int )
(declare-var main@%_2_0 Int )
(declare-var main@%_3_0 Bool )
(declare-var @__VERIFIER_nondet_int_0 Int )
(declare-var main@entry_0 Bool )
(declare-var main@_bb_0 Bool )
(declare-var main@_bb_1 Bool )
(declare-var main@.preheader2_0 Bool )
(declare-var main@.lr.ph7.preheader_0 Bool )
(declare-var main@.lr.ph7_0 Bool )
(declare-var main@%i.1.i6_0 Int )
(declare-var main@%k.0.i5_0 Int )
(declare-var main@%i.1.i6_1 Int )
(declare-var main@%k.0.i5_1 Int )
(declare-var main@.preheader1_0 Bool )
(declare-var main@%i.1.i.lcssa_0 Int )
(declare-var main@%k.0.i.lcssa_0 Int )
(declare-var main@%i.1.i.lcssa_1 Int )
(declare-var main@%k.0.i.lcssa_1 Int )
(declare-var main@_bb1_0 Bool )
(declare-var main@%_7_0 Int )
(declare-var main@%_8_0 Int )
(declare-var main@.lr.ph7_1 Bool )
(declare-var main@.preheader1.loopexit_0 Bool )
(declare-var main@%.lcssa13_0 Int )
(declare-var main@%.lcssa_0 Int )
(declare-var main@_bb1_1 Bool )
(declare-var main@.preheader_0 Bool )
(declare-var main@.lr.ph.preheader_0 Bool )
(declare-var main@.lr.ph_0 Bool )
(declare-var main@%j.0.i4_0 Int )
(declare-var main@%k.1.i3_0 Int )
(declare-var main@%j.0.i4_1 Int )
(declare-var main@%k.1.i3_1 Int )
(declare-var main@verifier.error_0 Bool )
(declare-var main@verifier.error.split_0 Bool )
(declare-var main@%_17_0 Int )
(declare-var main@%_18_0 Int )
(declare-var main@.lr.ph_1 Bool )
(declare-var main@verifier.error.loopexit_0 Bool )
(rule (verifier.error false false false))
(rule (verifier.error false true true))
(rule (verifier.error true false true))
(rule (verifier.error true true true))
(rule (main@entry @__VERIFIER_nondet_int_0))
(rule (=> (and (main@entry @__VERIFIER_nondet_int_0)
         true
         (=> main@_bb_0 (and main@_bb_0 main@entry_0))
         main@_bb_0)
    (main@_bb @__VERIFIER_nondet_int_0)))
(rule (=> (and (main@_bb @__VERIFIER_nondet_int_0)
         true
         (= main@%_1_0 @__VERIFIER_nondet_int_0)
         (= main@%_3_0 (= main@%_2_0 0))
         (=> main@_bb_1 (and main@_bb_1 main@_bb_0))
         main@_bb_1
         (=> (and main@_bb_1 main@_bb_0) (not main@%_3_0)))
    (main@_bb @__VERIFIER_nondet_int_0)))
(rule (let ((a!1 (and (main@_bb @__VERIFIER_nondet_int_0)
                true
                (= main@%_1_0 @__VERIFIER_nondet_int_0)
                (= main@%_3_0 (= main@%_2_0 0))
                (=> main@.preheader2_0 (and main@.preheader2_0 main@_bb_0))
                (=> (and main@.preheader2_0 main@_bb_0) main@%_3_0)
                (=> main@.preheader2_0 (= main@%_4_0 @__VERIFIER_nondet_int_0))
                (=> main@.preheader2_0 (= main@%_6_0 (= main@%_5_0 0)))
                (=> main@.lr.ph7.preheader_0
                    (and main@.lr.ph7.preheader_0 main@.preheader2_0))
                (=> (and main@.lr.ph7.preheader_0 main@.preheader2_0)
                    (not main@%_6_0))
                (=> main@.lr.ph7_0
                    (and main@.lr.ph7_0 main@.lr.ph7.preheader_0))
                main@.lr.ph7_0
                (=> (and main@.lr.ph7_0 main@.lr.ph7.preheader_0)
                    (= main@%i.1.i6_0 0))
                (=> (and main@.lr.ph7_0 main@.lr.ph7.preheader_0)
                    (= main@%k.0.i5_0 0))
                (=> (and main@.lr.ph7_0 main@.lr.ph7.preheader_0)
                    (= main@%i.1.i6_1 main@%i.1.i6_0))
                (=> (and main@.lr.ph7_0 main@.lr.ph7.preheader_0)
                    (= main@%k.0.i5_1 main@%k.0.i5_0)))))
  (=> a!1 (main@.lr.ph7 @__VERIFIER_nondet_int_0 main@%i.1.i6_1 main@%k.0.i5_1))))
(rule (let ((a!1 (and (main@_bb @__VERIFIER_nondet_int_0)
                true
                (= main@%_1_0 @__VERIFIER_nondet_int_0)
                (= main@%_3_0 (= main@%_2_0 0))
                (=> main@.preheader2_0 (and main@.preheader2_0 main@_bb_0))
                (=> (and main@.preheader2_0 main@_bb_0) main@%_3_0)
                (=> main@.preheader2_0 (= main@%_4_0 @__VERIFIER_nondet_int_0))
                (=> main@.preheader2_0 (= main@%_6_0 (= main@%_5_0 0)))
                (=> main@.preheader1_0
                    (and main@.preheader1_0 main@.preheader2_0))
                (=> (and main@.preheader1_0 main@.preheader2_0) main@%_6_0)
                (=> (and main@.preheader1_0 main@.preheader2_0)
                    (= main@%i.1.i.lcssa_0 0))
                (=> (and main@.preheader1_0 main@.preheader2_0)
                    (= main@%k.0.i.lcssa_0 0))
                (=> (and main@.preheader1_0 main@.preheader2_0)
                    (= main@%i.1.i.lcssa_1 main@%i.1.i.lcssa_0))
                (=> (and main@.preheader1_0 main@.preheader2_0)
                    (= main@%k.0.i.lcssa_1 main@%k.0.i.lcssa_0))
                (=> main@_bb1_0 (and main@_bb1_0 main@.preheader1_0))
                main@_bb1_0)))
  (=> a!1
      (main@_bb1 main@%i.1.i.lcssa_1
                 main@%k.0.i.lcssa_1
                 @__VERIFIER_nondet_int_0))))
(rule (=> (and (main@.lr.ph7 @__VERIFIER_nondet_int_0 main@%i.1.i6_0 main@%k.0.i5_0)
         true
         (= main@%_7_0 (+ main@%i.1.i6_0 1))
         (= main@%_8_0 (+ main@%k.0.i5_0 1))
         (= main@%_9_0 @__VERIFIER_nondet_int_0)
         (= main@%_11_0 (= main@%_10_0 0))
         (=> main@.lr.ph7_1 (and main@.lr.ph7_1 main@.lr.ph7_0))
         main@.lr.ph7_1
         (=> (and main@.lr.ph7_1 main@.lr.ph7_0) (not main@%_11_0))
         (=> (and main@.lr.ph7_1 main@.lr.ph7_0) (= main@%i.1.i6_1 main@%_7_0))
         (=> (and main@.lr.ph7_1 main@.lr.ph7_0) (= main@%k.0.i5_1 main@%_8_0))
         (=> (and main@.lr.ph7_1 main@.lr.ph7_0)
             (= main@%i.1.i6_2 main@%i.1.i6_1))
         (=> (and main@.lr.ph7_1 main@.lr.ph7_0)
             (= main@%k.0.i5_2 main@%k.0.i5_1)))
    (main@.lr.ph7 @__VERIFIER_nondet_int_0 main@%i.1.i6_2 main@%k.0.i5_2)))
(rule (=> (and (main@.lr.ph7 @__VERIFIER_nondet_int_0 main@%i.1.i6_0 main@%k.0.i5_0)
         true
         (= main@%_7_0 (+ main@%i.1.i6_0 1))
         (= main@%_8_0 (+ main@%k.0.i5_0 1))
         (= main@%_9_0 @__VERIFIER_nondet_int_0)
         (= main@%_11_0 (= main@%_10_0 0))
         (=> main@.preheader1.loopexit_0
             (and main@.preheader1.loopexit_0 main@.lr.ph7_0))
         (=> (and main@.preheader1.loopexit_0 main@.lr.ph7_0) main@%_11_0)
         (=> (and main@.preheader1.loopexit_0 main@.lr.ph7_0)
             (= main@%.lcssa13_0 main@%_8_0))
         (=> (and main@.preheader1.loopexit_0 main@.lr.ph7_0)
             (= main@%.lcssa_0 main@%_7_0))
         (=> (and main@.preheader1.loopexit_0 main@.lr.ph7_0)
             (= main@%.lcssa13_1 main@%.lcssa13_0))
         (=> (and main@.preheader1.loopexit_0 main@.lr.ph7_0)
             (= main@%.lcssa_1 main@%.lcssa_0))
         (=> main@.preheader1_0
             (and main@.preheader1_0 main@.preheader1.loopexit_0))
         (=> (and main@.preheader1_0 main@.preheader1.loopexit_0)
             (= main@%i.1.i.lcssa_0 main@%.lcssa_1))
         (=> (and main@.preheader1_0 main@.preheader1.loopexit_0)
             (= main@%k.0.i.lcssa_0 main@%.lcssa13_1))
         (=> (and main@.preheader1_0 main@.preheader1.loopexit_0)
             (= main@%i.1.i.lcssa_1 main@%i.1.i.lcssa_0))
         (=> (and main@.preheader1_0 main@.preheader1.loopexit_0)
             (= main@%k.0.i.lcssa_1 main@%k.0.i.lcssa_0))
         (=> main@_bb1_0 (and main@_bb1_0 main@.preheader1_0))
         main@_bb1_0)
    (main@_bb1 main@%i.1.i.lcssa_1 main@%k.0.i.lcssa_1 @__VERIFIER_nondet_int_0)))
(rule (=> (and (main@_bb1 main@%i.1.i.lcssa_0
                    main@%k.0.i.lcssa_0
                    @__VERIFIER_nondet_int_0)
         true
         (= main@%_13_0 @__VERIFIER_nondet_int_0)
         (= main@%_15_0 (= main@%_14_0 0))
         (=> main@_bb1_1 (and main@_bb1_1 main@_bb1_0))
         main@_bb1_1
         (=> (and main@_bb1_1 main@_bb1_0) (not main@%_15_0)))
    (main@_bb1 main@%i.1.i.lcssa_0 main@%k.0.i.lcssa_0 @__VERIFIER_nondet_int_0)))
(rule (let ((a!1 (and (main@_bb1 main@%i.1.i.lcssa_0
                           main@%k.0.i.lcssa_0
                           @__VERIFIER_nondet_int_0)
                true
                (= main@%_13_0 @__VERIFIER_nondet_int_0)
                (= main@%_15_0 (= main@%_14_0 0))
                (=> main@.preheader_0 (and main@.preheader_0 main@_bb1_0))
                (=> (and main@.preheader_0 main@_bb1_0) main@%_15_0)
                (=> main@.preheader_0
                    (= main@%_16_0 (> main@%k.0.i.lcssa_0 (- 1))))
                (=> main@.lr.ph.preheader_0
                    (and main@.lr.ph.preheader_0 main@.preheader_0))
                (=> (and main@.lr.ph.preheader_0 main@.preheader_0) main@%_16_0)
                (=> main@.lr.ph_0 (and main@.lr.ph_0 main@.lr.ph.preheader_0))
                main@.lr.ph_0
                (=> (and main@.lr.ph_0 main@.lr.ph.preheader_0)
                    (= main@%j.0.i4_0 0))
                (=> (and main@.lr.ph_0 main@.lr.ph.preheader_0)
                    (= main@%k.1.i3_0 main@%k.0.i.lcssa_0))
                (=> (and main@.lr.ph_0 main@.lr.ph.preheader_0)
                    (= main@%j.0.i4_1 main@%j.0.i4_0))
                (=> (and main@.lr.ph_0 main@.lr.ph.preheader_0)
                    (= main@%k.1.i3_1 main@%k.1.i3_0)))))
  (=> a!1 (main@.lr.ph main@%k.1.i3_1 main@%j.0.i4_1 main@%i.1.i.lcssa_0))))
(rule (let ((a!1 (and (main@_bb1 main@%i.1.i.lcssa_0
                           main@%k.0.i.lcssa_0
                           @__VERIFIER_nondet_int_0)
                true
                (= main@%_13_0 @__VERIFIER_nondet_int_0)
                (= main@%_15_0 (= main@%_14_0 0))
                (=> main@.preheader_0 (and main@.preheader_0 main@_bb1_0))
                (=> (and main@.preheader_0 main@_bb1_0) main@%_15_0)
                (=> main@.preheader_0
                    (= main@%_16_0 (> main@%k.0.i.lcssa_0 (- 1))))
                (=> main@verifier.error_0
                    (and main@verifier.error_0 main@.preheader_0))
                (=> (and main@verifier.error_0 main@.preheader_0)
                    (not main@%_16_0))
                (=> main@verifier.error.split_0
                    (and main@verifier.error.split_0 main@verifier.error_0))
                main@verifier.error.split_0)))
  (=> a!1 main@verifier.error.split)))
(rule (=> (and (main@.lr.ph main@%k.1.i3_0 main@%j.0.i4_0 main@%i.1.i.lcssa_0)
         true
         (= main@%_17_0 (+ main@%k.1.i3_0 (- 1)))
         (= main@%_18_0 (+ main@%j.0.i4_0 1))
         (= main@%_19_0 (< main@%_18_0 main@%i.1.i.lcssa_0))
         main@%_19_0
         (= main@%_20_0 (> main@%k.1.i3_0 0))
         (=> main@.lr.ph_1 (and main@.lr.ph_1 main@.lr.ph_0))
         main@.lr.ph_1
         (=> (and main@.lr.ph_1 main@.lr.ph_0) main@%_20_0)
         (=> (and main@.lr.ph_1 main@.lr.ph_0) (= main@%j.0.i4_1 main@%_18_0))
         (=> (and main@.lr.ph_1 main@.lr.ph_0) (= main@%k.1.i3_1 main@%_17_0))
         (=> (and main@.lr.ph_1 main@.lr.ph_0)
             (= main@%j.0.i4_2 main@%j.0.i4_1))
         (=> (and main@.lr.ph_1 main@.lr.ph_0)
             (= main@%k.1.i3_2 main@%k.1.i3_1)))
    (main@.lr.ph main@%k.1.i3_2 main@%j.0.i4_2 main@%i.1.i.lcssa_0)))
(rule (=> (and (main@.lr.ph main@%k.1.i3_0 main@%j.0.i4_0 main@%i.1.i.lcssa_0)
         true
         (= main@%_17_0 (+ main@%k.1.i3_0 (- 1)))
         (= main@%_18_0 (+ main@%j.0.i4_0 1))
         (= main@%_19_0 (< main@%_18_0 main@%i.1.i.lcssa_0))
         main@%_19_0
         (= main@%_20_0 (> main@%k.1.i3_0 0))
         (=> main@verifier.error.loopexit_0
             (and main@verifier.error.loopexit_0 main@.lr.ph_0))
         (=> (and main@verifier.error.loopexit_0 main@.lr.ph_0)
             (not main@%_20_0))
         (=> main@verifier.error_0
             (and main@verifier.error_0 main@verifier.error.loopexit_0))
         (=> main@verifier.error.split_0
             (and main@verifier.error.split_0 main@verifier.error_0))
         main@verifier.error.split_0)
    main@verifier.error.split))
(query main@verifier.error.split)

