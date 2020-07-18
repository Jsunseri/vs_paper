#!/bin/bash
# pass in the top dir, which should contain the Vina docked files
dpath=${1}
files=($(find "${dpath}" -maxdepth 1 -mindepth 1 -type f -name "AID*_docked.sdf.gz") $(find "${dpath}" -maxdepth 1 -mindepth 1 -regex .*_inactive_[0-9]+.*_docked.sdf))
dirs=(dense crossdock_default2018 general_default2018)
# dirs=(dense)
numseeds=5
outfile="${dpath}"/sdsorter.summary

# remove the stale one if it exists
rm -f "${dpath}"/sdsorter.summary
paste <(printf "Rank Title Vina MW Target File RFScore-VS RFScore-4") <(for dir in "${dirs[@]}"; do for ((i=0;i<$numseeds;i++)); do for stype in CNNaffinity CNNscore; do printf "${dir}_seed${i}_${stype} "; done; done; done) > "${outfile}"

# subdirs for the CNN models should contain cnnrescore_${fname} files
for file in "${files[@]}"; do 
  # this is just a reasonable guess
  target=$(basename "$(dirname "${file}")")
  base_with_ext=$(basename "${file}")
  base=$(basename "${file}" .sdf)
  if [ "${base}" != "${base_with_ext}" ]; then
    # then it wasn't gzipped
    nlines=$(grep -c '\$\$\$\$' "${file}" | awk '{print $1}');
  else
    # then it was gzipped
    base=$(basename "${base}" .sdf.gz)
    nlines=$(zgrep -c '\$\$\$\$' "${file}" | awk '{print $1}');
  fi

  # alternate scoring models, sanity check number of poses
  rfscorevs=${dpath}/${base}_rfscorevs.csv
  fixed=${dpath}/${base}_rfscorevs_fixed.csv
  rflines=$(wc -l ${rfscorevs} | awk '{print $1}')
  let rflines=${rflines}-1
  if [ ${rflines} != ${nlines} ]; then 
    echo size mismatch for ${rfscorevs} and ${file}
  fi
  # RFscore-VS output has Windows EOL characters that mess everything up,
  # so let's take care of that 
  if [ $(grep -c $'\r' ${rfscorevs}) != 0 ]; then
    awk 'gsub(/\r/,""){print $0}' ${rfscorevs} > ${fixed}
    mv ${fixed} ${rfscorevs}
  fi

  rfscore4=${dpath}/${base}_rfscore4.csv
  if [ $(wc -l ${rfscore4} | awk '{print $1}') != ${nlines} ]; then 
    echo size mismatch for ${rfscore4} and ${file}
  fi

  for dir in ${dirs[@]}; do
    for ((seed=0;seed<5;seed++)); do
      if [ ${seed} == 0 ]; then
        method=${dir}; else
        method=${dir}_${seed}
      fi
      if [ $(zgrep -c '\$\$\$\$' "${dpath}/${method}/${base}_rescore.sdf.gz") != ${nlines} ]; then
        echo size mismatch for ${method} and ${file}
      fi
    done
  done

  paste <(sdsorter -print -omit-header ${file}) <( for ((i=0; i<$nlines; i++)); do echo "${target} ${base}"; done) <(tail -n+2 ${rfscorevs} | awk -F, '{print " ", $3}') <(cat ${rfscore4}) > "${dpath}"/out; for dir in "${dirs[@]}"; do for ((i=0;i<5;i++)); do if [ ${i} == 0 ]; then method=${dir}; else method=${dir}_${i}; fi; sdsorter -print -omit-header -keep-tag CNNaffinity -keep-tag CNNscore ${dpath}/${method}/${base}_rescore.sdf.gz | awk '{print $3,$4}' > "${dpath}"/tmp; paste "${dpath}"/out "${dpath}"/tmp > "${dpath}"/out1; mv "${dpath}"/out1 "${dpath}"/out; done; done; cat "${dpath}"/out >> "${dpath}"/sdsorter.summary
done

# clean that shit up what a mess
rm -f "${dpath}"/out
rm -f "${dpath}"/tmp
