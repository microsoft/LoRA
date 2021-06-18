#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import sys
import io
import json


with open(sys.argv[1], 'r', encoding='utf8') as reader, \
	 open(sys.argv[2], 'w', encoding='utf8') as writer :
	for line in reader:
		items = line.strip().split('||')
		context = items[0]
		completion = items[1].strip('\n')
		x = {}
		x['context'] = context #+ '||'
		x['completion'] = completion
		writer.write(json.dumps(x)+'\n')

