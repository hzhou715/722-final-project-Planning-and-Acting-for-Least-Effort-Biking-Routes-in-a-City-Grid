(define (domain grid-costs)
  (:requirements :typing :adl :action-costs)
  (:types cell)

  (:predicates
    (at ?c - cell)
    (adj ?from ?to - cell)
  )

  (:functions
    (move-cost ?c - cell)
    (total-cost)
  )

  (:action move
    :parameters (?from ?to - cell)
    :precondition (and
      (at ?from)
      (adj ?from ?to)
    )
    :effect (and
      (not (at ?from))
      (at ?to)
      (increase (total-cost) (move-cost ?to))
    )
  )
)
