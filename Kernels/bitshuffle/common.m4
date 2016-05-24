#/*
#* sort-sources.cpp
#* Part of 2d-treecodes
#*
#* Created and authored by Diego Rossinelli on 2015-11-25.
#* Copyright 2015. All rights reserved.
#*
#* Users are NOT authorized
#* to employ the present software for their own publications
#* before getting a written permission from the author of this file.
#*/

define(`forloop',
       `pushdef(`$1', `$2')_forloop(`$1', `$2', `$3', `$4')popdef(`$1')')

define(`_forloop',
       `$4`'ifelse($1, `$3', ,
		   `define(`$1', incr($1))_forloop(`$1', `$2', `$3', `$4')')')

define(`forrloop',
       `pushdef(`$1', `$2')_forrloop(`$1', `$2', `$3', `$4')popdef(`$1')')

define(`_forrloop',
       `$4`'ifelse($1, `$3', ,
		   `define(`$1', decr($1))_forrloop(`$1', `$2', `$3', `$4')')')

USAGE LUNROLL
$1 iteration variable
$2 iteration start
$3 iteration end
$4 body

define(LUNROLL, `forloop($1, $2, $3,`$4')')
define(RLUNROLL, `forrloop($1, $2, $3, `$4')')
define(`TMP', $1_$2)

define(`REDUCEL',`
ifelse(eval($# < 3), 1,, $2 = $1($2, $3);)' `ifelse(eval($# <= 3), 1,`',`REDUCEL($1, shift(shift(shift($*))))')')

define(`ODDREMOVE', `ifelse(eval($# <= 2), 1, ifelse(eval($# > 0),1,$1), `$1,ODDREMOVE(shift(shift($*)))')')

define(`REDUCE',`REDUCEL($*)' `ifelse(eval($# <= 3), 1, ,`
REDUCE($1, ODDREMOVE(shift($*)))')')

#example: REDUCE(`+=', s0, s1, s2, s3, s4, s5, s6, s7)
define(TILE, 8)
