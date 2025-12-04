import xml.etree.ElementTree as ET
import pathlib
import json

FEATURE_TYPE_TEXT = "Digital Currency Address - "
NAMESPACE = {'sdn': 'https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/ADVANCED_XML'}

POSSIBLE_ASSETS = ["XBT", "ETH", "XMR", "LTC", "ZEC", "DASH", "BTG", "ETC",
                   "BSV", "BCH", "XVG", "USDT", "XRP", "ARB", "BSC", "USDC",
                   "TRX"]

OUTPUT_FORMATS = ["TXT", "JSON"]

def feature_type_text(asset):
    return "Digital Currency Address - " + asset


def get_address_id(root, asset):
    feature_type = root.find(
        "sdn:ReferenceValueSets/sdn:FeatureTypeValues/*[.='{}']".format(feature_type_text(asset)), NAMESPACE)
    if feature_type == None:
        raise LookupError("No FeatureType with the name {} found".format(
            feature_type_text(asset)))
    address_id = feature_type.attrib["ID"]
    return address_id


def get_sanctioned_addresses(root, address_id):
    """returns a list of sanctioned addresses for the given address_id"""
    addresses = list()
    for feature in root.findall("sdn:DistinctParties//*[@FeatureTypeID='{}']".format(address_id), NAMESPACE):
        for version_detail in feature.findall(".//sdn:VersionDetail", NAMESPACE):
            addresses.append(version_detail.text)
    return addresses


def write_addresses(addresses, asset, output_formats, outpath):
    if "TXT" in output_formats:
        write_addresses_txt(addresses, asset, outpath)
    if "JSON" in output_formats:
        write_addresses_json(addresses, asset, outpath)


def write_addresses_txt(addresses, asset, outpath):
    with open("{}/sanctioned_addresses_{}.txt".format(outpath, asset), 'w') as out:
        for address in addresses:
            out.write(address+"\n")


def write_addresses_json(addresses, asset, outpath):
    with open("{}/sanctioned_addresses_{}.json".format(outpath, asset), 'w') as out:
        out.write(json.dumps(addresses, indent=2)+"\n")


def run(assets=None, sdn_path="data/datasets/sdn_advanced.xml", output_formats=None, outpath=pathlib.Path("./")):

    if assets is None:
        assets = [POSSIBLE_ASSETS[1]]
    elif isinstance(assets, str):
        assets = [assets]

    if output_formats is None:
        output_formats = [OUTPUT_FORMATS[0]]
    elif isinstance(output_formats, str):
        output_formats = [output_formats]

    tree = ET.parse(sdn_path)
    root = tree.getroot()

    if not isinstance(outpath, pathlib.Path):
        outpath = pathlib.Path(outpath)

    for asset in assets:
        address_id = get_address_id(root, asset)
        addresses = get_sanctioned_addresses(root, address_id)

        addresses = list(dict.fromkeys(addresses).keys())
        addresses.sort()

        write_addresses(addresses, asset, output_formats, outpath)


if __name__ == "__main__":
    run()