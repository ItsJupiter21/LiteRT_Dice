# LiteRT_Dice

Dice recognition Neural network? i guess!

## Goals

Make a classification model capable of reliably recognising dice values,

Must support:

- D4's (table POV)
- D6's with pips/numbers
- D8's
- D10/D100's
- D12's
- D20's

And should be relatively easy to add new dice.

Reliability must be decent enough to detect inconsistencies in the dice itself.

Performance must be more than 1fps running on a Pi zero 2w for all of them.

## Non-Goals

- Recognise Dice type
- One model for everything(every dice will be a seperate model)
- Object detection(the input image will be a standalone dice.)

## Tech?

I'm probably going to develop everything in tensforflow, and then use LiteRT on the pi.
